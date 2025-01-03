import numpy as np
import cv2
import tiffslide
from shapely.geometry import Point, Polygon

# Function to get the annotation ID
def get_annotation_id(gc, item_id, annotation_name):
    annotations = gc.get(f'annotation', parameters={'itemId': item_id.strip()})
    for annotation in annotations:
        if annotation['annotation']['name'] == annotation_name:
            return annotation['_id']
    raise ValueError(f"Annotation with name '{annotation_name}' not found for item ID {item_id}.")

# Function to get the annotation elements
def get_annotation_elements(gc, annotation_id):
    annotation = gc.get(f'annotation/{annotation_id}')
    if 'elements' in annotation['annotation']:
        return annotation['annotation']['elements']
    else:
        raise ValueError(f"'elements' key not found in annotation with ID {annotation_id}")

# Function to process the point annotation elements
def process_point_annotation_elements(elements):
    points = []
    for element in elements:
        if element['type'] == 'point':
            points.append(Point(element['center'][:2]))
    return points

# Function to process the annotation elements
def process_annotation_elements(elements):
    labels = {}
    for element in elements:
        label = element.get('label', {}).get('value', "unknown")
        if label:
            if label not in labels:
                labels[label] = []
            labels[label].append(np.array(element['points'])[:, :2])

    return labels

# Function to get the artery polygons
def find_closest_artery_to_point(point, artery_polygons):
    closest_artery = None
    min_distance = float('inf')
    
    for artery_poly in artery_polygons:
        artery_polygon = Polygon(artery_poly)
        
        if artery_polygon.contains(point):
            return artery_poly
        
        distance = point.distance(artery_polygon)
        if distance < min_distance:
            min_distance = distance
            closest_artery = artery_poly
            
    return closest_artery

# Function to save the cropped artery image
def save_cropped_artery_image(svs_file_path, artery_poly, output_cropped_path, save_flag, margin=10):
    artery_polygon = Polygon(artery_poly)

    slide = tiffslide.TiffSlide(svs_file_path)
    slide_dims = slide.level_dimensions[0]

    x_min, y_min, x_max, y_max = artery_polygon.bounds

    x_min = max(0, int(x_min) - margin)
    y_min = max(0, int(y_min) - margin)
    x_max = min(slide_dims[0], int(x_max) + margin)
    y_max = min(slide_dims[1], int(y_max) + margin)

    region = slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min))
    region_np = np.array(region.convert("RGB"))
    if save_flag:
        cv2.imwrite(output_cropped_path, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR))
    return x_min, y_min, x_max, y_max



