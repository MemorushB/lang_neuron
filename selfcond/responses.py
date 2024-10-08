#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import pathlib
import pickle
import typing as t
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from selfcond.models import (
    TorchModel,
    ResponseInfo,
    processors_per_model,
    MODEL_INPUT_FIELDS,
    LABELS_FIELD,
)

# Replace print statements with logging
logging.basicConfig(level=logging.INFO)

def save_batch(batch: t.Dict[str, np.ndarray], batch_index: int, save_path: pathlib.Path) -> None:
    with (save_path / f"{batch_index:05d}.pkl").open("wb") as fp:
        pickle.dump(batch, fp)


def cache_responses(
    model: TorchModel,
    dataset: Dataset,
    response_infos: t.List[ResponseInfo],
    batch_size: int,
    save_path: pathlib.Path,
) -> None:
    """
    Caches the responses of a ``model`` as serialized files in ``save_path``.
    Responses are read from the tensors described in ``response_infos``.

    Args:
        model: A ``TorchModel`` that allows reading intermediate responses.
        dataset: The dataset (torch) to be fed to the model.
        response_infos: A list of response infos.
        batch_size: The inference batch size.
        save_path: Where to save the responses.
    """
    assert batch_size == 1, "Batch size should always be 1."

    def _concatenate_data(x):
        new_batch = dict()
        for key in x[0].keys():
            if isinstance(x[0][key], str):
                new_batch[key] = np.array([x[idx][key] for idx in range(len(x))])
            else:
                new_batch[key] = torch.tensor([x[idx][key] for idx in range(len(x))])
        return new_batch

    save_path.mkdir(parents=True, exist_ok=True)
    process_fn_list = processors_per_model(model)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_concatenate_data
    )
    for i, batch in tqdm(enumerate(data_loader), desc="Caching inference"):
        input_batch = {k: v for k, v in batch.items() if k in MODEL_INPUT_FIELDS}
        
        print("========test========")
        num_effective_token = torch.sum(input_batch["attention_mask"][0])
        num_effective_token = num_effective_token.detach().cpu().item()
        print(num_effective_token)
        print(input_batch["attention_mask"].shape)
        print(input_batch["input_ids"].shape)

        input_batch["attention_mask"] = input_batch["attention_mask"][:, :num_effective_token]
        input_batch["input_ids"] = input_batch["input_ids"][:, :num_effective_token]
        print(input_batch["attention_mask"].shape)
        print(input_batch["input_ids"].shape)
        # Until here

        
        generated_texts = model.generate_output(inputs=input_batch)
        keywords = batch['keywords']
        
        # Only save responses that contain the keyword (correct answer)
        should_save = [keyword.lower() in text.lower() for text, keyword in zip(generated_texts, keywords)]
        
        if not any(should_save):
            continue
        
        # Run inference to get responses
        response_batch = model.run_inference(
            inputs=input_batch, outputs={ri.name for ri in response_infos}
        )

        # Filter the responses based on should_save
        for key in response_batch.keys():
            response_batch[key] = response_batch[key][should_save]

        # Filter labels if needed
        if LABELS_FIELD in batch:
            response_batch[LABELS_FIELD] = batch[LABELS_FIELD][should_save].detach().cpu().numpy()

        # Apply processing functions to ensure consistent shapes
        for process_fn in process_fn_list:
            response_batch = process_fn(response_batch)

        # Save the filtered and processed batch
        save_batch(batch=response_batch, batch_index=i, save_path=save_path)
        


def read_responses_from_cached(
    cached_dir: pathlib.Path, concept: str, verbose: bool = False
) -> t.Tuple[t.Dict[str, np.ndarray], t.Optional[np.ndarray], t.Set[str]]:
    """
    Reads model responses stored on disk. The responses are stored pickled, one file per batch,
    structured as follows:
    * Responses accessible as a dictionary `{layer_name: responses}`. For example, `responses['layer_1']` is a
      multidimensional array of floats.

    Args:
        cached_dir: Directory with *.pkl files.
        concept: Concept for which the labels will be read.
        verbose: Verbosity flag.

    Returns:
        data: dict of {layer_name: np.ndarray} with the responses of all the batches concatenated and transposed.
              Each layer response is of shape [units, total_samples].
        labels: np.ndarray with the labels of all the data points.
        response_names: The names of the layers that have produced the responses.
    """
    # Initialize variables
    data: t.Dict[str, np.ndarray] = {}
    labels: t.List[float] = []
    response_names: t.Set[str] = set()
    labels_name = LABELS_FIELD

    # Get all cached files
    all_files = sorted(list(cached_dir.glob("*.pkl")))
    if not all_files:
        raise RuntimeError(f"No cached response files found in {cached_dir}")

    # Data structure to hold responses temporarily
    data_as_lists: t.Dict[str, t.List[np.ndarray]] = {}

    # Read each cached batch
    for file_name in tqdm(all_files, total=len(all_files), desc=f"Loading {concept}"):
        with file_name.open("rb") as fp:
            response_batch = pickle.load(fp)

            # Collect response names from the first batch
            if not response_names:
                response_names = set(response_batch.keys()) - {labels_name}

            # Process each key in the response batch
            for key, value in response_batch.items():
                if key == labels_name:
                    # Ensure value is a list or array
                    labels.extend(value if isinstance(value, (list, np.ndarray)) else [value])
                else:
                    if key not in data_as_lists:
                        data_as_lists[key] = []
                    data_as_lists[key].append(value)

    # Concatenate and transpose the data for each layer
    for key in data_as_lists.keys():
        try:
            # Concatenate along axis 0
            concatenated = np.concatenate(data_as_lists[key], axis=0)
            # Transpose to get shape [units, total_samples]
            data[key] = concatenated.transpose()

            if verbose:
                logging.info(f"Layer {key}, shape {data[key].shape}")
        except Exception as e:
            logging.error(f"Error processing layer {key}: {e}")
            continue

    # Convert labels to a numpy array if they exist
    labels_array = np.array(labels) if labels else None

    return data, labels_array, response_names
