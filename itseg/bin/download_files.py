from girder_client import GirderClient
from typing import List
from itseg.config import DEFAULT_ANNOTATIONS_REQ,ARTERIES_FEATURE_FILE
from itseg.utils import get_local_file_path, get_file_name,getFiles



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
def check_and_download_files(item_id:str, file_id:str,  gc: GirderClient, annotation_list=None) -> str:
    if not annotation_list:
        annotation_list = DEFAULT_ANNOTATIONS_REQ
    
    if not check_annotations(annotation_list=annotation_list, item_id=item_id, gc=gc):
        raise Exception(f"Missing the annotations required for the plugin {annotation_list}")
    
    local_path = get_local_file_path()
    file_name = get_file_name(gc, fileid=file_id)
    full_file_path = f"{local_path}/{file_name}"
    gc.downloadFile(fileId=file_id, path=full_file_path)
    download_feature_files(gc,itemId=item_id, feature_fnames=[ARTERIES_FEATURE_FILE])
    return full_file_path


def download_feature_files(gc:GirderClient, itemId: str, feature_fids: List=None, feature_fnames: List = None):
    if not feature_fids and not feature_fnames:
        return
    
    if not feature_fids:
        feature_fids = []
    
    if not feature_fnames:
        feature_fnames = []
    
    files = getFiles(itemId=itemId, gc=gc)
    local_path = get_local_file_path()
    
    def check_if_exists(filename:str) -> bool:
        for file in files:
            if file['name'].strip() == filename:
                return file['_id']
        return None
    
    for feature_file_name in feature_fnames:
        fid = check_if_exists(feature_file_name)
        if fid:
            feature_fids.append(fid)

    for id in feature_fids:
        path = f"{local_path}/{get_file_name(gc, id)}"
        gc.downloadFile(id, path=path)

        