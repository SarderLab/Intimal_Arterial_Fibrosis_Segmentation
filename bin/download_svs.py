import os
import csv
import girder_client

# Function to download SVS files from given CSV file
def download_svs_files(csv_path, folder_name, wsi_dir, gc):
    wsi_dir_path = os.path.join(wsi_dir, folder_name)
    os.makedirs(wsi_dir_path, exist_ok=True)
    
    count = 0
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            item_name = row['Item Name'].strip()
            item_id = row['Item ID'].strip()

            svs_file_path = os.path.join(wsi_dir_path, item_name)
            if os.path.exists(svs_file_path):
                count += 1
                print(f"SVS file already exists: {svs_file_path}")
                continue

            try:
                item_files = gc.get(f'item/{item_id}/files')
                svs_file_id = None

                for file_info in item_files:
                    if file_info['name'] == item_name:
                        svs_file_id = file_info['_id']
                        break

                if not svs_file_id:
                    print(f"No SVS file found for item {item_id} with name {item_name}")
                    continue

                print(f"Downloading SVS file: {item_name}")
                count += 1
                gc.downloadFile(svs_file_id, svs_file_path)

            except girder_client.HttpError as e:
                print(f"HTTP error occurred while processing item {item_id}: {e}")
            except Exception as e:
                print(f"Error occurred while processing item {item_id}: {e}")
    return count