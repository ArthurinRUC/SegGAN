import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from utils import gen_info, label_from_file


class XVIEW_DATA(Dataset):
    def __init__(self, data_dir):
        super(XVIEW_DATA, self).__init__()
        self.data_dir = data_dir
        info = data_dir + "/info.csv"
        if not os.path.exists(info):
            gen_info(data_dir)
        self.info = pd.read_csv(info)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        img_pre = read_image(
            self.info.iloc[index]["png_pre"])  # (3, 1024, 1024)
        img_pre = img_pre/255
        img_post = read_image(
            self.info.iloc[index]["png_post"])  # (3, 1024, 1024)
        img_post = img_post/255
        label_pre = label_from_file(
            self.info.iloc[index]["json_pre"])  # (1, 1024, 1024)
        label_post = label_from_file(
            self.info.iloc[index]["json_post"])  # (4, 1024, 1024)
        return img_pre, img_post, label_pre, label_post


if __name__ == "__main__":
    a = XVIEW_DATA("../data")
    b = a.__getitem__(1)
    for i in b:
        print(i.shape)
