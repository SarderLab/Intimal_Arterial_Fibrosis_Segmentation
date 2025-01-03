import girder_client
import pandas as pd

from bin.utils import create_dir

def check_annotation(gc, item_id, item_name, annotation_reqs = ["Intima_test", "arteries/arterioles"]):
    annotations_url = 'annotation'
    
    try:
        response = gc.get(annotations_url, parameters={'itemId': item_id})
        annotations = [annotation['annotation']['name'].strip() for annotation in response]
        if "Intima_test" not in annotations:
            print(f"Item {item_name} does not have Intima_test annotation")
        
        if "arteries/arterioles" not in annotations:
            print(f"Item {item_name} does not have arteries/arterioles annotation")
            
        return all([annotation_req in annotations for annotation_req in annotation_reqs])
    
    except Exception as e:
        print(f'Error retrieving annotations for item {item_id}: {e}')

def process_folder_annotations(gc, folder_id):
    annotation_reqs = ["Intima_test", "arteries/arterioles"]
    try:
        files = list(gc.listItem(folder_id))
        folder_name = gc.getFolder(folder_id)['name']
        data = []

        for file in files:
            item_name = file['name']
            item_id = file['_id']
            print(f"Processing item with ID: {item_id}\t{item_name}")
            if check_annotation(gc, item_id, item_name, annotation_reqs):
                data.append([item_name, item_id])

        print(f"Found {len(data)} items with annotation {annotation_reqs} in folder {folder_id}")
        df = pd.DataFrame(data, columns=['Item Name', 'Item ID'])
        create_dir('public')
        folder_path = f'public/{folder_name}.csv'
        df.to_csv(folder_path, index=False)
        return folder_name, folder_path

    except Exception as e:
        print(f'Error processing folder {folder_id}: {e}')
