import os
import copy
import json
from .config import DEFAULT_DOCKER_TMP_PATH, TMPDIR_ENV_NAME, COLLECTION_NAME,OP_FOLDER_NAME,JSON_ANNOTATION_TEMPLATE, PREDICTIONS_FOLDER_NAME, IMAGES_FOLDER_NAME
from girder_client import GirderClient

def get_local_file_path():
    #Try to access mnt location (Specific to Docker dsa deployment at UF) #TODO
    filepath = ''
    if os.path.exists(DEFAULT_DOCKER_TMP_PATH):
        filepath = '{}/{}'.format(DEFAULT_DOCKER_TMP_PATH, os.listdir(DEFAULT_DOCKER_TMP_PATH)[0])
    #DSA's tmp file location is specified with this env value. 
    elif os.getenv(TMPDIR_ENV_NAME,None):
        filepath = os.getenv(TMPDIR_ENV_NAME)
    else:
        raise Exception("Please set the temporary folder filepath ENV variable")
    return filepath

def getItemId(gc:GirderClient, file_id: str) -> str:
    """This function is used to get the corresponding itemId for a fileId on Girder.

    Args:
        gc (GirderClient): The girder-client object that is authentaicated. 
        file_id (str): The file_id of the file 

    Returns:
        str: Return the ItemId of the given File.
    """
    if not gc:
        raise ValueError("Please pass the gc object to access girder API")

    if not file_id:
        raise ValueError("File ID cannot be empty")
    
    try:
        response = gc.get(f'/file/{file_id}')
        return response['itemId']
    except Exception as e:
        raise e
    
def get_file_name(gc:GirderClient,fileid:str) -> str:
    """This function is used to return the filename for a given fileid

    Args:
        gc (GirderClient): The girderClient object for interacting with Girder API
        fileid (str): The file id of the file for which name is requested. 

    Returns:
        str: Return the filename on Girder for the requested fileid
    """
    try:
        if not fileid:
            raise ValueError("FileId cannot be empty")
        response = gc.get(f'/file/{fileid}')
        if not response.get('name'):
            raise ValueError("File Name is empty string")
        return response.get('name')
    except Exception as e:
        raise e

def getFiles(gc:GirderClient, itemId:str) -> list[dict]:
    """This function returns all the files that are present in the files section of an item on Girder.

    Args:
        gc (GirderClient): The girderClient object for interacting with Girder API
        itemId (str): The itemId of the item for which this information is requested. 

    Returns:
        list[dict]: Returns a list of JSON objects where each object corresponds to complete information about a file listed under that item.
    """
    endpoint = f'/item/{itemId}/files'
    try:
        if not itemId or not gc:
            raise ValueError("Please pass the ItemId and the girder client object")
        response = gc.get(endpoint)
        return response
    except Exception as e:
        raise e

def get_folderId(gc: GirderClient, collectionName: str = None) -> str:
    """This function retrieves the folderId where the output pngs will be stored so they can be added and referenced from the annotation file. 
    By default, it looks for a collection with Name IntemaSegmentation. If it doesn't find that collection, it creates it. 
    After this, it checks inside this collection, for a folder output. It creates it if absent and returns the folderId

    Args:
        gc (GirderClient): The Girder Client object to interact with the Girder API
        collectionName (str, optional): Change the Collection name where the output masks will be stored. Defaults to None.

    Returns:
        str: Return the folderID where the resuting masks will be stored. 
    """
    if not gc:
        raise ValueError("Please pass the girder client object")
    if not collectionName:
        collectionName = COLLECTION_NAME

    try:
        collectionEndpoint = f'/collection?text={collectionName}'
        collection = gc.get(collectionEndpoint)
        if not collection:
            collection = gc.createCollection(COLLECTION_NAME, public=True)
        
        if isinstance(collection, list):
            collectionId = collection[0]['_id']
        else:
            collectionId = collection["_id"]

        if not collectionId:
            raise Exception(f"Unable to create Collection {COLLECTION_NAME}")
        folder = gc.get(path='folder', parameters={"parentType": "collection",
                                                   "parentId": collectionId,
                                                   "name": OP_FOLDER_NAME})
        if not folder:
            folder = gc.createFolder(parentId=collectionId,parentType='collection', name=OP_FOLDER_NAME, public=True)
        if isinstance(folder, list):
            return folder[0]["_id"]
        return folder["_id"]
    
    except Exception as e:
        raise e
    
def generateAnnotationJSON(girder_items:list,box_coords_dict:dict) -> dict:
    """This function is used to generate the Annotation data required for the Annotation layer for displaying the results of IntemaSegmentation. 

    Args:
        girder_items (list): A list of (imagename, girderId) where imagename is the name of the result mask and girderId is the corresponding id on Girder for the same image.
        box_coords_dict (dict): Provides the original dimensions of the Result image.

    Returns:
        dict: Returns a dict that will be used for the annotation file.
    """
    json_data = copy.deepcopy(JSON_ANNOTATION_TEMPLATE)
    elements = []
    for image, itemId in girder_items:
        annotation_item = {
            "type": "image",
            "opacity": 0.5,
            "hasAlpha": False
        }
        bbox = box_coords_dict[image]
        if not bbox:
            continue
        width = bbox["xmin"]
        height = bbox["ymin"]
        annotation_item["girderId"] = itemId
        annotation_item["transform"] = {
            "xoffset": width,
            "yoffset": height
        }
        elements.append(annotation_item)
    json_data["elements"] = elements
    return json_data
    
        

def uploadIntemaAnnotations(gc:GirderClient,item_id:str,dataset_path:str,box_coords_dict:dict) -> None:
    """This function uploads a copy of the output masks under the files section of the Image, an annotation layer under the annotations section and also uploads a copy 
    of the large-image style png file that is being referenced by the annotation layer. 

    Args:
        gc (GirderClient): The girder client object to interact with the Girder API
        item_id (str): The itemID of the svs image on Girder
        dataset_path (str): The basepath for accessing the local result files. 
        box_coords_dict (dict): A dict containing the original dimensions of all the resulting masks. 

    """
    if not dataset_path:
        raise ValueError("Please pass the info file containing the crop image information")
    
    pred_path = os.path.join(dataset_path,PREDICTIONS_FOLDER_NAME)
    images = [item for item in os.listdir(pred_path) if item.endswith('.png')]
    
    cropped_img_path = os.path.join(dataset_path, IMAGES_FOLDER_NAME)
    cropped_images = [item for item in os.listdir(cropped_img_path) if item.endswith('.png')]
    
    girder_items = []

    for idx, image in enumerate(cropped_images):
        image_path = f"{cropped_img_path}/{image}"
        uploadImages(gc,item_id,file_path=image_path,filename=f"original_crop_{idx}")
    
    for image in images:
        image_path = f"{pred_path}/{image}"
        folderId = get_folderId(gc=gc)
        #Upload a copy of these to the files section for other dependent plugins. 
        # gc.uploadFileToItem(itemId=item_id, filepath=image_path)
        uploadImages(gc,item_id,file_path=image_path)

        #Upload another copy for generating annotation
        try:
            output = gc.uploadFileToFolder(folderId=folderId,filepath=image_path)
            itemId = output.get('itemId')
            #Pngs are incompatible if used for annotations. This below function converts them to a large-image format to make them compatible. 
            gc.post(f'/item/{itemId}/tiles')
            girder_items.append((image,itemId))
        except Exception as e:
            print(e)
    
    json_data = generateAnnotationJSON(girder_items,box_coords_dict)
    gc.post(path='annotation',parameters={"itemId":item_id}, data=json.dumps(json_data))


def uploadImages(gc: GirderClient, item_id:str, file_path:str,filename=None) -> None:
    try:
        if not gc or not item_id or not file_path:
            raise Exception(f"Please pass the parameters for file upload - (gc, item_id, file_path)")
        gc.uploadFileToItem(itemId=item_id, filepath=file_path, filename=filename)
    except Exception as e:
        raise e
    
    