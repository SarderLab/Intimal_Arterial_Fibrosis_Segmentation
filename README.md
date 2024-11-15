# Intimal Fibrosis Computation Project

## Introduction
This project focuses on segmenting and calculating intima measurements in WSI images using three main scripts. The `train` script allows users to train a custom UNET model, while `evaluate` assesses model performance on test data. The core `inference` script processes WSI images by downloading `.svs` files based on a user-provided Folder ID, filters images containing **intima** and **artery** annotations, and crops regions of interest. It then applies the trained model to generate predictions and calculates key metrics like **Intimal Thickness** and **Stenosis Ratio** through computer vision operations.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/SarderLab/Intimal_Arterial_Fibrosis_Segmentation
cd Intimal_Arterial_Fibrosis_Segmentation
```

2. Set up the environment:
(Add specific installation steps)

## Scripts
- `train.py`: This script is the training script that the user can use to train over a certain user specified images and ground truth masks. The script assumes that the user specifies cropped PNG images and equivalent ground truth masks which are aligned to the intima in the image. This the crucial for the model to perform accordingly and avoid incorrect segmentation. The training script requires the user to specify many arguments which are given below as follows
    - `--epochs`: Number of epochs to run the model
    - `--folds`: Number of folds to generate over the dataset. This training script splits the training data into specified number of folds. of training and validation groups.
    - `--batch_size`: The batch size to use when training.
    - `--lr`: The initial learning rate for model to use when updating weights. The fold models use a scheduler over this learning rate, but the main model over the complete data keeps the same learning rate all over.
    - `--seed`: The user can specify a seed for dataset spliting and random generation. This helps in reproducing results for research purposes.
    - `--split`: Specifies a fraction/floating point used to split the data between training and testing. The default value is given as **0.15**.

    The script is run in the following given steps. This is given for understanding the flow of learning.

    1. The script starts with setting the seed, preparing the device, creating directories if not existing, etc which are initial necessary opeartions required for the script to run.
    2. This is followed by the `create_dataset` function which splits the data in the specified path according to the given `--split` and stores the splits into the `./dataset` folder.
    3. The script then initializes a `torch` Dataset from the training split. Subsequently, `--folds` are generated from the Dataset and according `torch` DataLoaders are initialized. The script uses the `nn.BCELoss` for comparing the model to the ground truth and 0.5 thresholding method for generating predictions.
    4. The algorithm requires (`--folds` + 1) UNET models to be intialized. `--folds` models are used over the respective number of splits. The one extra model is trained over the entire data.
    5. This model is saved at every point the best average metrics of `--folds` models are observed over `--epochs` number of epochs. All these models are saved in `./checpoints` directory under a folder named by the UNETWithAttention followed by a unique identifier for the model.
    6. The average metrics over the fold models and the metrics of the main model are all logged and plotted over complete training process. These images are saved in a directory called `./files`.
    7. Lastly, we require only the best models for each metrics used while training. Thus we run the `keep_best_models` function to only retain the best model for each metric.

- `evaluate.py`: The evaluation script goes hand in hand with the training script. The evaluation script uses the test set split formulated in the training script to generate evaluations. The results of the evaluation are logged when the script is run, but users can also view the results stored in csv format in the `./results/` directory. The csv file helps use see the performance of the model on each image from the test set. Like the training, the evaluate script does not require many arguments to be specified. 
    - `--seed`: The user can specify a seed for dataset spliting and random generation. This helps in reproducing results for research purposes.
    - `--uid`: The model's unique identifier string which can be specified. The respective model is fetched from the `./checkpoints` directory.

    The evaluate script runs all the model checkpoints present in the specified model's uid directory. This helps user evaluate the average metrics, observe the results from the `csv` files and make a hcoice for which model performs the best.

- `inference.py`: The final script is the inference script. The user again need to specify many arguments necessary for running inference any the user specified data using user specified model. The script takes a directory of WSI images, crops the correctly annotated WSI images into png images and stores them. The model is then used to generate predictions over these images and finally a `csv` is generated summarizing the requried computations for each image. The following arguments are expected from the inference script:
    - `--api_url`: This requires the API_URL that Girder Client uses to the filter and fetch data.
    - `--token`: User's private token for accessing the API URL.
    - `--folder_id`: Takes in the folder ID which is unique to the each folder in the API URLs file system.
    - `--artery_cropping_margin`: Margin(in pixels) used to surround the annotated intima before cropping.
    - `--working_dir`: This is the path to the working directory where the WSI images are downloaded, cropped images and predictions are stored.
    - `--model_path`: The model path is required for the model checkpoint to be loaded and intialized.
    - `--model_save_type`: The best metrics according to which the respective checkpoint will be loaded.
    - `--model_uid`: Specifies the model's unique identifier which is used to load the model.

    The inference script is followed a series of predefined steps which are run in the given order for generating predictions and computation results.
    1. The script starts with initializing a Girder Client `gc` object which is used to fetch WSI files, annotations, etc.
    2. The `process_folder_annotation` function is then called which collects all the WSI files in the give `folder_id` with the specified annotation present in them and store the names and ids in a `csv` file names after the specific `folder_name` peratining to the `folder_id` using `gc` client.
    3. The `download_svs` function downloads all these `.svs` files in the specifed directory path `working_dir/folder_name/WSI`.
    4. The `process_csv_and_generate_crops` function is then called to take in all the WSI images downloaded, find point annotations for each, crop the images along with the specified `artery_margin`(pixels) around it. The cropped images are stored in the `working_dir/dataset/folder_name/images` path.
    5. Once the cropped images are generated, the script prepares the specified model checkpoint from the `model_path` using the `model_save_type` and `model_uid`. The model is then used to generate predictions over all the cropped images and the predictions are stored in `working_dir/dataset/folder_name/predictions` path.
    6. The `evaluate_csv` function is the last function in the script which takes all the predictions for each cropped artery image, and performs morphological operation like **contourDetection**, **distanceTransform** and **skeletonize** to calculate the necessary computations.
    7. The results of this are saved in a `csv` which is stored in `working_dir/dataset/folder_name/intimal_pipeline_results.csv`

## Bugs and Assumptions
- Assumptions:
    - The `train.py` assumes that the user is specifiying a path to the cropped images and respective ground truth masks. This assumption is done as the model can only perform correctly if the crops and masks are exactly aligned in resolution.
    - The `inference.py` assumes that the user is giving folder ID with point annotated WSI images with annotations name `Intima_test` and `arteries/arterioles`.

