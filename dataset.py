import os
import kagglehub
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path if dataset_path else self.download_from_kaggle()
        
    def show_dataset_structure(self):
        print(os.listdir(self.dataset_path))
        df = pd.read_csv(os.path.join(dataset_path, "HAM10000_metadata.csv"))
        
        print(df.head())
    
    def download_from_kaggle(self):
        try:
            path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
            print("Path to dataset files:", path)
            return path
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            raise
            
        print("Path to dataset files:", path)
    

dataset_path = "/Users/arcsinx/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2"
image_dataset = ImageDataset(dataset_path=dataset_path)

image_dataset.show_dataset_structure()