import os
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import girder_client

from ctk_cli import CLIArgumentParser

from shapely.geometry import Polygon
from itseg.bin.download_files import check_and_download_files
from bin.artery_cropping import get_annotation_id, get_annotation_elements, process_point_annotation_elements, process_annotation_elements, find_closest_artery_to_point, save_cropped_artery_image
from bin.get_prediction import evaluate
from bin.evaluate_csv import evaluateCSV
from bin.utils import create_dir
from itseg.utils import getItemId,get_local_file_path
from itseg.config import INTIMA_ANNOTATION_NAME, ARTERIES_ANNOTATION_NAME,BBOX_INFO_FILE,IMAGES_FOLDER_NAME


def process_csv_and_generate_crops(gc: girder_client.GirderClient, svs_file_path: str, item_id: str, dataset_dir:str, annotation_name_intima:str, annotation_name_arteries:str, artery_margin, save_flag=True):
    box_coords = []
    box_coords_dict = {}
    item_name = os.path.basename(svs_file_path)
    try:
        artery_polygons = []
        annotation_id_arteries = get_annotation_id(gc, item_id, annotation_name_arteries)
        elements_arteries = get_annotation_elements(gc, annotation_id_arteries)
        labels_arteries = process_annotation_elements(elements_arteries)
        artery_polygons.extend([Polygon(poly) for polys in labels_arteries.values() for poly in polys if len(poly) >= 4])

        annotation_id_intima = get_annotation_id(gc, item_id, annotation_name_intima)
        elements_intima = get_annotation_elements(gc, annotation_id_intima)
        points_intima = process_point_annotation_elements(elements_intima)

        for point_index, point in enumerate(points_intima):
            closest_artery_poly = find_closest_artery_to_point(point, artery_polygons)

            if closest_artery_poly:
                base_filename = f"{os.path.splitext(item_name)[0]}_point{point_index}"
                create_dir(os.path.join(dataset_dir, IMAGES_FOLDER_NAME))
                output_cropped_path = os.path.join(dataset_dir, IMAGES_FOLDER_NAME, f"{base_filename}_cropped.png")

                xmin, ymin, xmax, ymax = save_cropped_artery_image(svs_file_path, closest_artery_poly, output_cropped_path, save_flag, margin=artery_margin)
                box_coords.append([item_name, point_index, xmin, ymin, xmax, ymax])
                box_coords_dict[f"{base_filename}_cropped.png"] = {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax":xmax,
                    "ymax":ymax
                }
            else:
                print(f"No artery found close to the point at index {point_index} for item {item_name}.")

    except ValueError as e:
        print(f"Error processing {item_id}: {e}")

    # Write box coordinates to CSV
    output_csv_path = os.path.join(dataset_dir,BBOX_INFO_FILE)
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Slide Name', 'Cropped Artery Index', 'Xmin', 'Ymin', 'Xmax', 'Ymax'])
        writer.writerows(box_coords)
    return output_csv_path, box_coords_dict

if __name__ == '__main__':
    args = CLIArgumentParser().parse_args()
    
    ip_img_fid = args.file_id
    artery_margin = int(args.artery_cropping_margin)
    gc = None
    if args.girderApiUrl and args.girderToken:
        gc = girder_client.GirderClient(apiUrl=str(args.girderApiUrl))
        gc.setToken(str(args.girderToken))
    else:
        raise ValueError("API URL and Token are required.")

    ip_img_itemId = getItemId(gc, file_id=ip_img_fid)
    svs_file_path = check_and_download_files(item_id=ip_img_itemId, file_id=ip_img_fid, gc=gc)
    dataset_dir = get_local_file_path()
    
    model_file_path = args.model_file
    if not model_file_path:
        raise ValueError("Model file cannot be empty")

    output_csv_path, box_coords_dict = process_csv_and_generate_crops(gc, svs_file_path=svs_file_path, item_id=ip_img_itemId, dataset_dir=dataset_dir, annotation_name_intima=INTIMA_ANNOTATION_NAME, annotation_name_arteries=ARTERIES_ANNOTATION_NAME, artery_margin=artery_margin)
    pred_path = evaluate(gc, ip_img_itemId,  dataset_dir, model_file_path=model_file_path, box_coords_dict=box_coords_dict)
    evaluateCSV(gc, ip_img_itemId, output_csv_path, svs_file_path, dataset_dir)