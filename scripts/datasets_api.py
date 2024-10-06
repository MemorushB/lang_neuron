import requests
import json
import time

def fetch_conceptnet_relations(relations, language='en', limit_per_request=1000, total_limit=None):
    base_url = 'https://api.conceptnet.io/query'
    results = {}
    for rel_name in relations:
        print(f"Fetching relation: {rel_name}")
        relation_uri = f"/r/{rel_name}"
        offset = 0
        total_fetched = 0
        pairs = []
        while True:
            params = {
                'rel': relation_uri,
                'other': f'/c/{language}',
                'limit': limit_per_request,
                'offset': offset
            }
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching data for relation {rel_name}: {response.status_code}")
                print(response.text)
                break
            data = response.json()
            edges = data.get('edges', [])
            if not edges:
                break
            for edge in edges:
                start_lang = edge['start'].get('language', '')
                end_lang = edge['end'].get('language', '')
                # Ensure both start and end are in the desired language
                if start_lang == language and end_lang == language:
                    start_label = edge['start'].get('label', '')
                    end_label = edge['end'].get('label', '')
                    pairs.append((start_label, end_label))
                    total_fetched += 1
                    if total_limit and total_fetched >= total_limit:
                        break
            if total_limit and total_fetched >= total_limit:
                break
            # Check if there is a next page
            next_page_url = data.get('view', {}).get('nextPage', '')
            if next_page_url:
                # Extract the offset from the next_page_url
                offset_param = next_page_url.split('offset=')[1]
                offset = int(offset_param.split('&')[0])
            else:
                break
            # Optional: Add delay to respect API rate limits
            time.sleep(1)
        results[rel_name] = pairs
        print(f"Fetched {len(pairs)} pairs for relation {rel_name}")
    return results

if __name__ == "__main__":
    # List of relations to fetch
    relations = ['Antonym', 'FormOf', 'Synonym', 'HasSubevent', 'HasFirstSubevent', 'HasLastSubevent', 'IsA', 'PartOf', 'HasA']
    # Set the desired number of items to fetch per request (max 1000)
    limit_per_request = 500
    # Set the total number of items to fetch per relation (optional)
    total_limit = 2000  # Set to None to fetch all available data
    # Fetch the relations
    relations_data = fetch_conceptnet_relations(relations, language='en', limit_per_request=limit_per_request, total_limit=total_limit)

    # Print the number of pairs fetched for each relation
    for rel, pairs in relations_data.items():
        print(f"\nRelation: {rel}, Number of pairs: {len(pairs)}")
        # Print some sample pairs
        print("Sample pairs:")
        for pair in pairs[:5]:
            print(f"  {pair[0]} - {pair[1]}")

    # Optionally, save the data to JSON files
    for rel, pairs in relations_data.items():
        filename = f"datasets/{rel}_pairs.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(pairs)} pairs to {filename}")