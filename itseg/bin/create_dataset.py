import os
import shutil
import cv2
import bin.utils as utils
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import  HorizontalFlip, CoarseDropout, RandomBrightnessContrast

# Load dataset from the given path
def load_dataset(path):
    images = sorted(glob(os.path.join(path, 'images', '*')))
    masks = sorted(glob(os.path.join(path, 'masks', '*')))

    return images, masks

# Split the dataset into train and test
def split_dataset(images, masks, split = 0.15, seed = 42):
    split_size = int(len(images) * split)

    train_x, test_x = train_test_split(images, test_size=split_size, random_state=seed)
    train_y, test_y = train_test_split(masks, test_size=split_size, random_state=seed)

    return (train_x, train_y), (test_x, test_y)

# Save the dataset
def save_dataset(images, masks, save_dir, augment=False):
    for x, y in tqdm(zip(images, masks), total=len(images)):
        file = x.split('/')[-1]
        name = file.split('.')[0]
        ext = file.split('.')[-1]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            aug = CoarseDropout(p=1.0, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = RandomBrightnessContrast(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug_x = [x, x1, x2, x3]
            aug_y = [y, y1, y2, y3]
        else:
            aug_x = [x]
            aug_y = [y]

        idx = 0
        for ax, ay in zip(aug_x, aug_y):
            aug_name = f'{name}_{idx}.{ext}'

            save_image_path = os.path.join(save_dir, 'images', aug_name)
            save_mask_path = os.path.join(save_dir, 'masks', aug_name)

            cv2.imwrite(save_image_path, ax)
            cv2.imwrite(save_mask_path, ay)
            idx += 1

# Create dataset from the given path
def create_dataset(data_path, dataset_path, split, seed):
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    for item in ['train', 'test']:
        utils.create_dir(os.path.join(dataset_path, item, 'images'))
        utils.create_dir(os.path.join(dataset_path, item, 'masks'))

    train_x, train_y = [], []
    test_x, test_y = [], []

    for folder in os.listdir(data_path):
        data_folder = os.path.join(data_path, folder)
        images, masks = load_dataset(data_folder)
        print(f'{data_folder}\t Images: {len(images)} - Masks: {len(masks)}')
        (tnx, tny), (tsx, tsy) = split_dataset(images, masks, split, seed)
        train_x.extend(tnx)
        train_y.extend(tny)
        test_x.extend(tsx)
        test_y.extend(tsy)

    if len(train_x) != len(train_y):
        raise ValueError(f'Number of images not equal to number of masks\nImages: {len(train_x)} - Masks: {len(train_y)}')
    
    if len(test_x) != len(test_y):
        raise ValueError(f'Number of images not equal to number of masks\nImages: {len(test_x)} - Masks: {len(test_y)}')
    
    save_dataset(train_x, train_y, os.path.join(dataset_path, 'train'), augment=True)
    save_dataset(test_x, test_y, os.path.join(dataset_path, 'test'))

