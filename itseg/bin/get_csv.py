from girder_client import GirderClient
from typing import List
from itseg.config import DEFAULT_ANNOTATIONS_REQ
from itseg.utils import get_local_file_path, get_file_name



def check_annotations(annotation_list:List[str], gc: GirderClient, item_id:str) -> bool:
    """This function is used to check whether all the required annotations are present before proceeding with next steps in the plugin.

    Args:
        annotation_list (list): A list of strings which specify the name of the required annotations. 
        gc (GirderClient): A gc object that allows to interact with the Girder API
        item_id (str): The item_id for the SVS file with which the annotations are associated. 

    Returns:
        bool: Returns true if all annotations required are present. Else, returns False. 
    """
    if not annotation_list:
        raise ValueError("Please provide atleast one annotation file to verify")
    
    annotation_endpoint = 'annotation'
    response = gc.get(annotation_endpoint, parameters={'itemId': item_id})
    annotations = [annotation['annotation']['name'].strip().lower() for annotation in response]
    for annotationName in annotation_list:
        if annotationName.lower() not in annotations:
            print(f'Annotation {annotationName} is not present on the slide. Please upload it and try again')
            return False
    return True

#Check whether both the annotation files are present
 # Download svs file. 
def check_and_download_file(item_id:str, file_id:str,  gc: GirderClient, annotation_list=None) -> str:
    if not annotation_list:
        annotation_list = DEFAULT_ANNOTATIONS_REQ
    
    if not check_annotations(annotation_list=annotation_list, item_id=item_id, gc=gc):
        raise Exception(f"Missing the annotations required for the plugin {annotation_list}")
    
    local_path = get_local_file_path()
    file_name = get_file_name(gc, fileid=file_id)
    full_file_path = f"{local_path}/{file_name}"
    gc.downloadFile(fileId=file_id, path=full_file_path)
    return full_file_path
    