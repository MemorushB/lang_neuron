import requests
import json

def fetch_conceptnet_relations(relations, language='en', limit=1000):
    """
    Fetch pairs from ConceptNet for the specified relations.

    Parameters:
    - relations: List of relation names (e.g., 'Antonym', 'Synonym')
    - language: Language code (default 'en' for English)
    - limit: Maximum number of results to fetch per API request

    Returns:
    - Dictionary with relation names as keys and lists of (start_label, end_label) tuples as values
    """
    base_url = 'http://api.conceptnet.io/query'
    results = {}
    for rel_name in relations:
        print(f"Fetching relation: {rel_name}")
        relation_uri = f"/r/{rel_name}"
        offset = 0
        pairs = []
        while True:
            params = {
                'rel': relation_uri,
                'start': f'/c/{language}/',
                'end': f'/c/{language}/',
                'limit': limit,
                'offset': offset
            }
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching data for relation {rel_name}: {response.status_code}")
                break
            data = response.json()
            edges = data.get('edges', [])
            if not edges:
                break
            for edge in edges:
                start_label = edge['start'].get('label', '')
                end_label = edge['end'].get('label', '')
                pairs.append((start_label, end_label))
            # Check if there is a next page
            if 'view' in data and 'nextPage' in data['view']:
                offset += limit
            else:
                break
        results[rel_name] = pairs
        print(f"Fetched {len(pairs)} pairs for relation {rel_name}")
    return results

if __name__ == "__main__":
    # List of relations to fetch
    relations = ['Antonym', 'FormOf', 'Synonym', 'HasSubevent', 'HasFirstSubevent', 'HasLastSubevent', 'IsA', 'PartOf', 'HasA']
    # Fetch the relations
    relations_data = fetch_conceptnet_relations(relations, language='en', limit=1000)

    # Print the number of pairs fetched for each relation
    for rel, pairs in relations_data.items():
        print(f"\nRelation: {rel}, Number of pairs: {len(pairs)}")
        # Print some sample pairs
        print("Sample pairs:")
        for pair in pairs[:5]:
            print(f"  {pair[0]} - {pair[1]}")

    # Optionally, save the data to JSON files
    for rel, pairs in relations_data.items():
        filename = f"{rel}_pairs.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(pairs)} pairs to {filename}")