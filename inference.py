import os
import csv
import girder_client
import argparse

from shapely.geometry import Polygon
from bin.get_csv import process_folder_annotations
from bin.download_svs import download_svs_files
from bin.artery_cropping import get_annotation_id, get_annotation_elements, process_point_annotation_elements, process_annotation_elements, find_closest_artery_to_point, save_cropped_artery_image
from bin.get_prediction import evaluate
from bin.evaluate_csv import evaluate_csv
from bin.utils import create_dir


def process_csv_and_generate_crops(gc, csv_path, folder_name, wsi_dir, dataset_dir, annotation_name_intima, annotation_name_arteries, artery_margin, save_flag):
    box_coords = []
    create_dir(os.path.join(dataset_dir, folder_name))

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            item_name = row['Item Name'].strip()
            item_id = row['Item ID'].strip()

            svs_file_path = os.path.join(wsi_dir, folder_name, item_name)
            if not os.path.exists(svs_file_path):
                print(f"SVS file not found: {svs_file_path}")
                continue

            try:
                annotation_id_intima = get_annotation_id(gc, item_id, annotation_name_intima)

                artery_polygons = []
                artery_found = False

                for artery_name in annotation_name_arteries:
                    try:
                        annotation_id_arteries = get_annotation_id(gc, item_id, artery_name)
                        elements_arteries = get_annotation_elements(gc, annotation_id_arteries)
                        labels_arteries = process_annotation_elements(elements_arteries)

                        artery_polygons.extend([Polygon(poly) for polys in labels_arteries.values() for poly in polys if len(poly) >= 4])
                        artery_found = True
                    except ValueError as e:
                        print(f"Error processing {item_name}: {e}")
                        continue

                if not artery_found:
                    print(f"No artery annotation found for either name '{annotation_name_arteries[0]}' or '{annotation_name_arteries[1]}' for item {item_name}. Skipping...")
                    continue

                if not artery_polygons:
                    print(f"No artery polygons found for item {item_name}. Skipping...")
                    continue

                elements_intima = get_annotation_elements(gc, annotation_id_intima)
                points_intima = process_point_annotation_elements(elements_intima)

                for point_index, point in enumerate(points_intima):
                    closest_artery_poly = find_closest_artery_to_point(point, artery_polygons)

                    if closest_artery_poly:
                        base_filename = f"{os.path.splitext(item_name)[0]}_point{point_index}"
                        create_dir(os.path.join(dataset_dir, folder_name, 'images'))
                        output_cropped_path = os.path.join(dataset_dir, folder_name, 'images', f"{base_filename}_cropped.png")

                        xmin, ymin, x_max, ymax = save_cropped_artery_image(svs_file_path, closest_artery_poly, output_cropped_path, save_flag, margin=artery_margin)
                        box_coords.append([item_name, point_index, xmin, ymin, x_max, ymax])
                        print(f"{idx + 1}: Processing point at index {point_index} for item {item_name}...")
                    else:
                        print(f"No artery found close to the point at index {point_index} for item {item_name}.")

            except ValueError as e:
                print(f"Error processing {item_name}: {e}")

    # Write box coordinates to CSV
    print(f"Processed and cropped {len(box_coords)} images.")
    output_csv_path = os.path.join("public", f'box_coordinates.csv')
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Slide Name', 'Cropped Artery Index', 'Xmin', 'Ymin', 'Xmax', 'Ymax'])
        writer.writerows(box_coords)
    return output_csv_path, folder_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", help="API URL", required=True)
    parser.add_argument("--token", help="API Token (User Specific)", required=True)
    parser.add_argument("--folder_id", help="Folder ID", required=True)
    parser.add_argument("--artery_cropping_margin", help="Margin for artery cropping", required=False, default=30)
    parser.add_argument("--working_dir", help="Working Directory: Directory to store WSI images, cropped images and predictions", required=True)
    parser.add_argument("--model_path", help="Path to trained and tested model", required=True, default='checkpoints/')
    parser.add_argument("--model_save_type", help="Which checkpoint to use for model: loss/accuracy/adc", required=True, default='loss')
    parser.add_argument("--model_uid", help="Model Unique Identifier", required=True)
    args = parser.parse_args()
    
    folder_id = str(args.folder_id)
    artery_margin = int(args.artery_cropping_margin)
    working_dir = str(args.working_dir)
    feature_path = '/blue/pinaki.sarder/ujwalaguttikonda/KPMP_features/features/'

    if args.api_url and args.token:
        gc = girder_client.GirderClient(apiUrl=str(args.api_url))
        gc.setToken(str(args.token))
    else:
        raise ValueError("API URL and Token are required.")

    annotation_name_intima = "Intima_test"
    annotation_name_arteries = ["arteries/arterioles", " arteries/arterioles"]
    folder_name, csv_path = process_folder_annotations(gc, folder_id)

    wsi_dir = os.path.join(working_dir, 'WSI')
    dataset_dir = os.path.join(working_dir, 'dataset')
    wsi_count = download_svs_files(csv_path, folder_name, wsi_dir, gc)
    print(f"Downloaded {wsi_count} WSI files.")

    checkpoint_params = {
        'uid': str(args.model_uid),
        'val': str(args.model_save_type),
        'path': str(args.model_path),
    }
    save_flag=True

    output_csv_path, folder_name = process_csv_and_generate_crops(gc, csv_path, folder_name, wsi_dir, dataset_dir, annotation_name_intima, annotation_name_arteries, artery_margin, save_flag)
    pred_path = evaluate(os.path.join(dataset_dir, folder_name), checkpoint_params, save_flag)
    evaluate_csv(output_csv_path, os.path.join(feature_path, folder_name), os.path.join(dataset_dir, folder_name))