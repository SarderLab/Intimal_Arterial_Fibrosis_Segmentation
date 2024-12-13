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
        self.image_dir = path
        self.transforms = transforms
        
        self.images = os.listdir(self.image_dir)
        self.pre_normalize = None
        self.totensor = T.ToTensor()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IntimaDataset.WIDTH, IntimaDataset.HEIGHT), interpolation = cv2.INTER_AREA)
        image = image / 255.0

        image = self.totensor(image)
        return image, image_name