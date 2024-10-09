import csv
import json

def extract_relations(file_path, relations, language='en'):
    """
    Extract specified relations from the ConceptNet dataset.

    Parameters:
    - file_path: Path to the ConceptNet CSV file.
    - relations: Set of relation names to extract (e.g., {'Antonym', 'Synonym'}).
    - language: Language code to filter concepts (default is 'en' for English).

    Returns:
    - A dictionary with relation names as keys and lists of (start_label, end_label) tuples as values.
    """
    relation_prefix = '/r/'
    concept_prefix = f'/c/{language}/'

    # Initialize a dictionary to hold the results
    results = {rel: [] for rel in relations}

    # Open the ConceptNet CSV file
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')

        for row in reader:
            # Unpack the fields
            if len(row) != 6:
                continue  # Skip malformed rows
            uri, rel, start, end, weight, data = row

            # Extract the relation name
            rel_name = rel[len(relation_prefix):]

            # Check if the relation is one we're interested in
            if rel_name not in relations:
                continue

            # Check if both start and end concepts are in the specified language
            if not (start.startswith(concept_prefix) and end.startswith(concept_prefix)):
                continue

            # Extract the labels from the URIs
            start_label = start[len(concept_prefix):].split('/')[0].replace('_', ' ')
            end_label = end[len(concept_prefix):].split('/')[0].replace('_', ' ')

            # Append the pair to the appropriate list in the results
            results[rel_name].append((start_label, end_label))

    return results

if __name__ == "__main__":
    # Specify the path to your ConceptNet CSV file
    file_path = 'datasets/conceptnet-assertions-5.7.0.csv'

    # Define the relations you want to extract
    relations = {
        'Antonym',
        'FormOf',
        'Synonym',
        'HasSubevent',
        'HasFirstSubevent',
        'HasLastSubevent',
        'IsA',
        'PartOf',
        'HasA'
    }

    # Extract the relations
    relations_data = extract_relations(file_path, relations, language='en')

    # Print the number of pairs extracted for each relation
    for rel, pairs in relations_data.items():
        print(f"Relation: {rel}, Number of pairs: {len(pairs)}")
        # Optionally, print some sample pairs
        print("Sample pairs:")
        for pair in pairs[:5]:
            print(f"  {pair[0]} - {pair[1]}")
        print()

    # Optionally, save each relation's pairs to separate JSON files
    for rel, pairs in relations_data.items():
        filename = f"{rel}_pairs.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(pairs)} pairs to {filename}")