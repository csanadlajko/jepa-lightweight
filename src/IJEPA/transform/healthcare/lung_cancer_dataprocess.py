import os
from PIL import Image
from torch.utils.data import Dataset

class LungCancerDataset:

    """
    Lung cancer dataset object obtained from Kaggle by Biplob Dey.\n
    [Source] https://www.kaggle.com/datasets/biplobdey/lung-and-colon-cancer
    """

    def __init__(self, input_dir: str, transform=None):
        self.input_dir = input_dir
        self.sub_dirs = os.listdir(input_dir) ## benign, adenocarcinoma and agressive cell
        self.transform = transform

        self.lc_class_map = {
            "aca": 0, ## Lung Adenocarcinoma - cancerous cells of the lung
            "bnt": 1, ## Lung Benign Tissue - healthy lung tissues
            "scc": 2 ## Lung Squamous Cell Carcinoma - aggressive lung cancer type
        }
        
        self.images = [
            f ## file name
            for sub_dir in self.sub_dirs ## in sub dirs
            if "lung" in sub_dir ## which are containing lung cancer images
            for f in os.listdir(os.path.join(input_dir, sub_dir)) ## for file in that folder
        ]
        self.labels = [self.lc_class_map[f.split("_")[1]] for f in self.images]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        label: int = int(self.labels[idx])
        sub_folder_name: str = "_".join(self.images[idx].split("_")[:2])
        image_full_path: str = os.path.join(self.input_dir, sub_folder_name, self.images[idx])
        image = Image.open(image_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, label