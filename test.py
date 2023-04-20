from utils import label_from_file
from pprint import pprint
import torch

from torchvision import transforms

def set_color(a:torch.Tensor, color:tuple[int,int,int])->torch.Tensor:
    return torch.stack([i*a for i in color], dim=0)

img = label_from_file("../data/labels/hurricane-harvey_00000177_pre_disaster.json") * 255

toPIL = transforms.ToPILImage()

pic = toPIL(img)
pic.save("loc.png")

img = label_from_file("../data/labels/hurricane-harvey_00000177_post_disaster.json")
new_img = torch.zeros((3,1024,1024))
a = set_color(img[0], (142,207,201))
b = set_color(img[1], (130,176,210))
c = set_color(img[2], (255,190,122))
d = set_color(img[3], (250,127,111))
new_img = a+b+c+d
pic = toPIL(new_img)
pic.save("new_img.png")