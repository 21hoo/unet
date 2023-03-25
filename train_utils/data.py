import os

from torch.utils.data import Dataset
from utils import keep_image_size_open
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path:str, training=bool) -> None:
        super().__init__()
        self.flag = "training" if training else "test"
        self.path = os.path.join(path, self.flag)
        self.name = os.listdir(os.path.join(self.path, "images")) # 只有文件的名字不包含路劲

    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        images_name = self.name[index] # xxx.tif
        images_path = os.path.join(self.path, "images", images_name)
        mask_path = os.path.join(self.path, "mask", images_name.split(".")[0]+"_mask.gif")
        images_im = keep_image_size_open(images_path)
        mask_im = keep_image_size_open(mask_path)
        return transform(images_im), transform(mask_im)
    
if __name__ == '__main__':
    data = MyDataset("/root/workspace/U-Net/data/DRIVE")
    print(data[0][0].shape)
