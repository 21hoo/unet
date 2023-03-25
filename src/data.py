import os

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, path:str) -> None:
        super().__init__()
        self.path = path
        self.name = os.listdir(os.path.join(self.path, "iamges"))

    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        images_name = self.name[index] # xxx.tif
        images_path = os.path.join(self.path, "images", images_name)
        mask_path = os.path.join(self.path, "mask", images_name.split(".")[0]+"_mask.gif")