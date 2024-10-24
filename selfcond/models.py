#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
import typing as t
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoModelForPreTraining, AutoConfig, LlamaForCausalLM, XGLMForCausalLM, BloomForCausalLM

MODEL_INPUT_FIELDS = ["input_ids", "attention_mask"]
LABELS_FIELD = "labels"


@dataclass(frozen=True)
class ResponseInfo:
    """
    Information about of a model's response.

    A response is the output tensor of a model operation (ie. layer or a module depending on your
    deep-learning framework). Note that an operation may have more than one response.
    """

    name: str
    """Name of the response."""

    dtype: np.dtype
    """Data type of the response."""

    shape: t.Tuple[t.Optional[int], ...]
    """Shape of the response. The first dimension will generally be `None`."""

    layer: "ResponseInfo.Layer"
    """Details about the layer that generates this response."""

    @dataclass(frozen=True)
    class Layer:
        """
        Class to hold information about the layer that generated the response.
        """

        name: str
        """Name of the layer that generated the response."""

        kind: str
        """Type of layer."""


class TorchModel:
    """
    Class wrapping a Pytorch model so that we can read intermediate responses from it.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        input_size: t.Mapping[str, t.Tuple],
        input_type: t.Mapping[str, torch.dtype],
        name: str,
        device: str = None,
        tokenizer=None,  # Add tokenizer parameter
    ) -> None:
        """
        Wraps a pytorch module to enable reading intermediate responses.
        Args:
            module: A pytorch nn.module holding the model to be wrapped.
            input_size: A dict with model input names as keys and the expected sizes as values.
            input_type: A dict with model input names as keys and the expected types as values.
            name: The model name according to Huggingface Transformers.
            device: A string that indicates where the model should run (cpu, cuda:0, etc...)
        """
        self.tokenizer = tokenizer  # Store the tokenizer
        self.name = name

        # Set the device
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Move the model to the specified device
        self._pytorch_module = module.to(self._device).eval()

        if set(input_size.keys()) != set(input_type.keys()):
            raise RuntimeError(
                "Model input keys for size and type must be the same."
                f"{input_size.keys()} != {input_type.keys()}."
            )

        self._forward_hooks: t.List[RemovableHandle] = []
        self._input_size: t.Mapping[str, t.Tuple] = input_size
        self._input_types: t.Mapping[str, torch.dtype] = input_type
        self._response_infos: t.List[ResponseInfo] = []
        self._compute_response_infos()

    @property
    def module(self) -> nn.Module:
        return self._pytorch_module

    def _compute_response_infos(self) -> None:
        def hook(module_name, module, module_input, module_output) -> None:
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            outputs = module_output if isinstance(module_output, (list, tuple)) else [module_output]

            for output_idx, o in enumerate(outputs):
                if o is None or type(o) is not torch.Tensor:
                    continue

                response_name = "{}:{}".format(module_name, output_idx)
                ri = ResponseInfo(
                    name=response_name,
                    dtype=o.dtype,
                    shape=(o.size())[1:],
                    layer=ResponseInfo.Layer(
                        name=module_name,
                        kind=class_name,
                    ),
                )

                self._response_infos.append(ri)

        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        hooks = []
        for module_name, module in self._pytorch_module.named_modules():
            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                continue

            if module == self._pytorch_module:
                continue

            hooks.append(module.register_forward_hook(partial(hook, module_name)))

        # perform inference
        self._perform_dummy_inference()

        # remove forward hooks
        for h in hooks:
            h.remove()

    def _perform_dummy_inference(self) -> None:
        arg_names = list(self._input_types.keys())
        # batch_size of 2 in case of batchnorm
        fixed_shaped_list: t.List[int] = [2]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x = {
            input_name: torch.rand(tuple(fixed_shaped_list + [*self._input_size[input_name]]))
            .type(self._input_types[input_name])
            .to(device)
            for input_name in arg_names
        }

        # make a forward pass
        with torch.no_grad():  # type: ignore
            self._pytorch_module(**x)
        
    def generate_output(self, inputs: t.Mapping[str, torch.Tensor], max_length: int = None) -> t.List[str]:
        # Move inputs to the correct device
        torch_inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Prepare input IDs and attention mask
        input_ids = torch_inputs['input_ids']
        attention_mask = torch_inputs.get('attention_mask', None)

        # Generate outputs
        with torch.no_grad():
            generated_ids = self._pytorch_module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length or input_ids.shape[1] + 50,
                do_sample=False
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    def get_response_infos(self) -> t.Iterable[ResponseInfo]:
        """
        Generate a list of :class:`ResponseInfo`s with the name, type and other information of each response.
        Returns:
            A list of :class:`ResponseInfo` objects.
        """
        return self._response_infos

    def _set_units_hook_wrapper(
        self,
        units: torch.Tensor,
        values: torch.Tensor,
        only_last_token: bool,
    ) -> t.Callable:
        assert len(units) == len(values), "The number of values must match the number of units."
        # assert units.dtype == torch.int64, "Unit indices must be int64."
        # assert values.dtype == torch.float32, "Values must be float32."

        def forward_hook(module, input, output):
            # Modify the output of the layer.
            if only_last_token:
                output[:, -1, units] = values.to(output.device)
            else:
                output[:, :, units] = values.to(output.device)
            return output

        return forward_hook

    def set_units_in_layer(
        self,
        layer_name: str,
        units: torch.Tensor,
        values: torch.Tensor,
        only_last_token: bool = False,
    ) -> None:
        """
        Registers forward hooks that will set the indexed ``units`` in ``layer`` with the ``values`` passed.

        Args:
            layer_name: The layer (Tensor) name to be modified.
            units: Indices to the units to be set.
            values: Values to set the units to.
            only_last_token: Set only the last token of the sentence. If False, all tokens are set.
        """
        layer_name = layer_name.replace(":0", "")
        for iterated_module_name, layer in self._pytorch_module.named_modules():
            if iterated_module_name == layer_name:
                handle = layer.register_forward_hook(
                    self._set_units_hook_wrapper(
                        units=units,
                        values=values,
                        only_last_token=only_last_token,
                    )
                )
                self._forward_hooks.append(handle)

    def restore_units(self):
        for h in self._forward_hooks:
            h.remove()
        self._forward_hooks.clear()

    def run_inference(
        self, inputs: t.Mapping[str, torch.Tensor], outputs: t.AbstractSet[str]
    ) -> t.Dict[str, np.ndarray]:
        """
         Run inference on a single batch of input data and return the responses the layers specified in ``outputs``.

         Note:
             This function runs inference upon being called.

         Args:
             inputs: A map of layer name to ``torch.Tensor``.
             outputs: Selects responses to be collected from the model after feeding the input batch through.

        Returns:
             A map of response names to ``np.ndarray`` containing the requested model responses
             to the input data.
        """
        a_key = list(inputs.keys())[0]
        torch_inputs: t.MutableMapping[str, torch.Tensor] = {}
        if isinstance(inputs[a_key][0], torch.Tensor):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch_inputs = {k: v.to(device) for k, v in inputs.items()}

        response_dict: t.Dict[str, t.Any] = {}

        def hook(module_name, module, module_input, module_output) -> None:  # type: ignore
            module_output = (
                module_output if isinstance(module_output, (list, tuple)) else [module_output]
            )

            for output_idx, o in enumerate(module_output):
                response_name = "{}:{}".format(module_name, output_idx)
                if response_name in outputs:
                    """
                    tensor = (
                        o.detach().numpy() if self._device == "cpu" else o.detach().cpu().numpy()
                    )
                    """
                    if o.dtype == torch.float32:
                        tensor = (
                            o.detach().cpu().numpy()
                        )
                    else:
                        tensor = (
                            o.detach().cpu().to(dtype=torch.float32).numpy()
                        )

                    response_dict[response_name] = tensor

        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        hooks = []

        for module_name, module in self._pytorch_module.named_modules():

            if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
                continue

            if module == self._pytorch_module:
                continue

            hooks.append(module.register_forward_hook(partial(hook, module_name)))

        # perform inference
        with torch.no_grad():  # type: ignore
            self._pytorch_module(**torch_inputs)

        # remove forward hooks
        for h in hooks:
            h.remove()

        return response_dict


class PytorchTransformersModel(TorchModel):
    """
    Class wrapping a HuggingFace Transformers model in a readable model.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: t.Optional[pathlib.Path],
        seq_len: int,
        device: str = None,
        tokenizer=None,
    ) -> None:
        # Load the model
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir)

        # Initialize the parent class
        super().__init__(
            module=transformers_class_from_name(model_name, cache_dir),
            input_size={"input_ids": (seq_len,), "attention_mask": (seq_len,)},
            input_type={"input_ids": torch.long, "attention_mask": torch.long},
            name=model_name,
            device=device,
            tokenizer=tokenizer
        )

def transformers_model_name_to_family(model_name: str) -> str:
    """
    Get the family of the model based on the model name, as defined in the Huggingface transformers repository.

    For example: `bert-base-cased` belongs to the family `bert`
    Args:
        model_name: The model name

    Returns:
        str: The family name

    """
    if model_name.startswith("bert"):
        return "bert"
    elif model_name.startswith("openai"):
        return "openai"
    elif model_name.startswith("gpt2"):
        return "gpt2"
    elif model_name.startswith("xlnet"):
        return "xlnet"
    elif model_name.startswith("xlm"):
        return "xlm"
    elif model_name.startswith("roberta"):
        return "roberta"
    elif model_name.startswith("distilbert"):
        return "distilbert"
    elif model_name.startswith("ctrl"):
        return "ctrl"
    elif "bloom" in model_name:
        return "bloom"
    elif "Llama-2" in model_name:
        return "Llama-2"
    elif "llama" in model_name:
        return "llama"
    elif "falcon" in model_name:
        return "falcon"
    elif "xglm" in model_name:
        return "xglm"
    else:
        raise NotImplementedError(f"Model name to type not considered: {model_name}")


def transformers_class_from_name(
    model_name: str, cache_dir: t.Optional[pathlib.Path] = None, rand_weights: bool = False #, clm: bool = False, bit: bool = False,
) -> nn.Module:
    """
    Obtain a model as pytorch nn.Module given a name (as defined in the Huggingface transformers repo)

    Args:
        model_name: The huggingface transformers model name
        cache_dir: Local cache dir
        rand_weights: Use random weights or pre-trained weights

    Returns:
        nn.Module: a Pytorch nn.Module.

    """
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if rand_weights:
            config = AutoConfig.from_pretrained(model_name)
            m = AutoModelForPreTraining.from_config(config)
        else:
            if 'Llama-2' in model_name:
                m = AutoModelForCausalLM.from_pretrained(model_name,
                                                         trust_remote_code=True,
                                                         cache_dir=cache_dir,
                                                         device_map="auto"
                                                         ).to(device)
            elif 'xglm' in model_name:
                m = XGLMForCausalLM.from_pretrained(model_name, 
                                                    cache_dir=cache_dir
                                                   ).to(device)
            elif 'bloom' in model_name:
                m = BloomForCausalLM.from_pretrained(model_name, 
                                                     cache_dir=cache_dir
                                                    ).to(device)
            else:
                raise ValueError("error! model_name is not properly defined.")
            
            try:
                print(m.hf_device_map)
            except:
                pass
    
    except OSError:
        raise NotImplementedError(f"Model {model_name} could not be loaded.")
    assert m is not None
    return m


def get_layer_regex(model_name: str) -> t.Optional[t.List[str]]:
    """
    Create regex for the layers of interest for different model families.
    These are the layers where expert units will be explored.

    Note:
        Only GPT2 family supported for now.

    Args:
        model_name: The requested model name.

    Returns:
        A list of strings with the layer names.

    """
    family = transformers_model_name_to_family(model_name)
    layer_types = None
    if family == "gpt2":
        layer_types = [
            "transformer.h.([0-9]|[0-9][0-9]).attn.c_attn",
            "transformer.h.([0-9]|[0-9][0-9]).attn.c_proj",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.c_fc",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.c_proj",
        ]
    elif family == "bloom":
        layer_types = [
            "transformer.h.([0-9]|[0-9][0-9]).self_attention.query_key_value",
            "transformer.h.([0-9]|[0-9][0-9]).self_attention.dense",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.dense_h_to_4h",
            "transformer.h.([0-9]|[0-9][0-9]).mlp.dense_4h_to_h",
        ]
    elif family in ["llama", "Llama-2"]:
        layer_types = [
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.q_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.k_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.v_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).self_attn.o_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.gate_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.down_proj",
            "transformer.layers.([0-9]|[0-9][0-9]).mlp.up_proj",
        ]
    if family == "xglm":
        layer_types = [
            "model.layers.([0-9]|[0-9][0-9]).self_attn.k_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.v_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.q_proj",
            "model.layers.([0-9]|[0-9][0-9]).self_attn.out_proj",
            "model.layers.([0-9]|[0-9][0-9]).fc1",
            "model.layers.([0-9]|[0-9][0-9]).fc2",
        ]
    # Extend to other model families here if needed
    return layer_types


def _print_responses(ri: t.List[ResponseInfo]) -> None:
    assert len(ri), "No responses selected"
    print(f"Found {len(ri)} responses from model.")
    for r in ri:
        print("\t", r.name, r.shape)


def _collect_responses_info_for_model(model: TorchModel, model_family: str) -> t.List[ResponseInfo]:
    mapping = {
        "gpt2": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "bloom": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "llama": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "Llama-2": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "falcon": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        "xglm": [
            ri
            for ri in model.get_response_infos()
            if ri.layer.kind in ["Conv1D", "BertLayerNorm", "Linear"]
            and len(ri.shape) in [2, 3]
            and "lm_head" not in ri.name
        ],
        # Extend to other models here
    }
    return mapping[model_family]


def collect_responses_info(model_name: str, model: TorchModel) -> t.List[ResponseInfo]:
    """
    Build the information required to read responses from model.

    Args:
        model_name: The model name
        model: A TorchModel

    Returns:
        Responses info

    """
    family = transformers_model_name_to_family(model_name)
    responses_info = _collect_responses_info_for_model(model, family)
    _print_responses(responses_info)
    return responses_info


def concatenate_responses(
    responses: t.Dict[str, np.ndarray],
    response_fields: t.Set[str],
    output_field: str,
    axis: int,
) -> t.Dict[str, np.ndarray]:
    data = [tensor for field, tensor in responses.items() if field in response_fields]
    responses[output_field] = np.concatenate(data, axis=axis)
    for field in response_fields:
        del responses[field]
    return responses


def pool_responses(
    responses: t.Dict[str, np.ndarray],
    response_fields: t.Optional[t.Set[str]],
    axis: int,
    pooling_type: str = "mean",
) -> t.Dict[str, np.ndarray]:
    assert pooling_type in ["mean", "sum", "max", "median", "min"]
    pooler_fn = getattr(np, pooling_type)
    fields = response_fields or responses.keys()
    for field in fields:
        print(f"Pooling field '{field}', shape {responses[field].shape}, axis {axis}")
        if responses[field].size == 0:
            print(f"Field '{field}' is empty. Skipping pooling.")
            continue
        if axis >= len(responses[field].shape) or axis < -len(responses[field].shape):
            print(f"Axis {axis} is out of bounds for array of shape {responses[field].shape}")
            continue  # Or adjust the axis accordingly
        responses[field] = pooler_fn(responses[field], axis=axis)
        print(f"After pooling, shape {responses[field].shape}")
    return responses


def processors_per_model(model: TorchModel) -> t.List[t.Callable]:
    #pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="min")]
    pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="mean")]
    #pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="median")]
    #pool_args: t.List[t.Dict] = [dict(response_fields=None, axis=1, pooling_type="max")]
    process_fns: t.List[t.Callable] = [partial(pool_responses, **args) for args in pool_args]
    return process_fns
