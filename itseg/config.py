INTIMA_ANNOTATION_NAME = 'intima_test'
ARTERIES_ANNOTATION_NAME = "arteries/arterioles"
DEFAULT_ANNOTATIONS_REQ = [INTIMA_ANNOTATION_NAME, ARTERIES_ANNOTATION_NAME]
DEFAULT_DOCKER_TMP_PATH = '/mnt/girder_worker'
TMPDIR_ENV_NAME = 'TMPDIR'
ARTERIES_FEATURE_FILE = 'arteriesarterioles_Features.xlsx'
BBOX_INFO_FILE = 'box_coordinates.csv'
COLLECTION_NAME = 'IntemaSegmenation'
OP_FOLDER_NAME = 'outputs'
PREDICTIONS_FOLDER_NAME = 'predictions'
IMAGES_FOLDER_NAME = 'images'
RESULT_XLSX_FILENAME = 'intimal_pipeline_results.xlsx'
INTIMA_XLSX_COLNAMES = ['Sr No', 'Slide Name', 'Cropped Artery Name','Cropped Artery Index','Xmin','Ymin','Xmax','Ymax','CompartmentID', 'Luminal Area(pix2)', 'Luminal Radius(pix)', 'Intimal Area(pix2)', 'Average Intimal Thickness(pix)', 'Outer Boundary Intimal Thickness(pix)', 'isMaskComplete', 'Stenosis Ratio(%)']

JSON_ANNOTATION_TEMPLATE = {
        "elements": [],
        "name": "Intema_Seg_output"
}