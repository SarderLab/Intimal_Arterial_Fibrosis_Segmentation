import os
from .config import DEFAULT_DOCKER_TMP_PATH, TMPDIR_ENV_NAME
from girder_client import GirderClient

def get_local_file_path():
    #Try to access mnt location (Specific to Docker dsa deployment at UF) #TODO
    filepath = ''
    if os.path.exists(DEFAULT_DOCKER_TMP_PATH):
        filepath = '{}/{}'.format('/mnt/girder_worker', os.listdir('/mnt/girder_worker')[0])
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
    
def get_file_name(gc,fileid:str) -> str:
    try:
        if not fileid:
            raise ValueError("FileId cannot be empty")
        response = gc.get(f'/file/{fileid}')
        if not response.get('name'):
            raise ValueError("File Name is empty string")
        return response.get('name')
    except Exception as e:
        raise e