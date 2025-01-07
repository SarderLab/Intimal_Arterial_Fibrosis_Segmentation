import os
from .config import DEFAULT_DOCKER_TMP_PATH, TMPDIR_ENV_NAME

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