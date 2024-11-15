import cv2
import os
import numpy as np
import uuid
from torchvision import transforms as T
from torch.utils.data import Dataset

# Dataset class for the Intima Segmentation Dataset
class IntimaDataset(Dataset):
    WIDTH = 800
    HEIGHT = 800
    
    def __init__(self, path, transforms=None) -> None:
        self.uid = uuid.uuid4().hex[:8]
        self.path = path
        self.image_dir = f"{self.path}/images"
        self.mask_dir = f"{self.path}/masks"
        self.transforms = transforms
        
        if len(os.listdir(self.image_dir)) != len(os.listdir(self.mask_dir)):
            raise ValueError("Number of images and masks do not match")
        
        self.images = os.listdir(self.image_dir)
        self.pre_normalize = None
        self.totensor = T.ToTensor()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IntimaDataset.WIDTH, IntimaDataset.HEIGHT), interpolation = cv2.INTER_AREA)
        image = image / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IntimaDataset.WIDTH, IntimaDataset.HEIGHT), interpolation = cv2.INTER_AREA)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)

        image = self.totensor(image)
        mask = self.totensor(mask)
        return image, mask, image_name