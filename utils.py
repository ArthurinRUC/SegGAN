import os
import json
import torch
import numpy as np
import pandas as pd
import cv2
from shapely import wkt
from shapely.geometry import Polygon
import torch.nn as nn
import torch.optim as op
from torch.nn import init

subtype_dict = {
    "destroyed": 4,
    "major-damage": 3,
    "minor-damage": 2,
    "no-damage": 1,
    "un-classified": 1
    # "no-building": 0
}


def mask_for_polygon(poly: Polygon, im_size: tuple[int, int] = (1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    def int_coords(x): return np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    cv2.fillPoly(img_mask, exteriors, color=1)
    return img_mask


def label_from_file(file_name: str, subtype_dict: dict = subtype_dict) -> torch.Tensor:
    with open(file_name) as f:
        im_size = (1024, 1024)
        json_info = json.load(f)
        
        if "pre" in file_name:
            res = np.zeros(im_size, np.uint8)  # (1024, 1024)
            for poly in json_info["features"]["xy"]:
                res += mask_for_polygon(wkt.loads(poly["wkt"]))
            res[res > 1] = 1
            res = np.expand_dims(res, axis=0)  # (1, 1024, 1024)
        elif "post" in file_name:
            res = np.zeros((4, *im_size), np.uint8)  # (4, 1024, 1024)
            for poly in json_info["features"]["xy"]:
                subtype = poly["properties"]["subtype"]
                res[subtype_dict[subtype] - 1
                    ] += mask_for_polygon(wkt.loads(poly["wkt"]))
            res[res > 1] = 1
        return torch.from_numpy(res)


def onehot_to_number(a: torch.Tensor) -> torch.Tensor:
    # a: (batch_size, 4, 1024, 1024)
    # "destroyed": 4,
    # "major-damage": 3,
    # "minor-damage": 2,
    # "no-damage": 1,
    # "un-classified": 1
    temp = torch.zeros((a.shape[0], 1, 1024, 1024), device=a.device)
    a = torch.cat([temp, a], dim=1)
    return torch.unsqueeze(torch.argmax(a, dim=1), 1)


def gen_info(data_dir: str) -> None:
    images = os.listdir(data_dir+"/images")
    image_pd = pd.DataFrame({"full_name": images})
    image_pd["id"] = image_pd["full_name"].apply(
        lambda x: x.split("_")[0]+x.split("_")[1])
    image_pd["status"] = image_pd["full_name"].apply(
        lambda x: x.split("_")[2])  # 提取是post还是pre
    image_pd_pre = image_pd[image_pd["status"] == "pre"]
    image_pd_post = image_pd[image_pd["status"] == "post"]
    image_pd = pd.merge(image_pd_pre, image_pd_post, on="id")
    image_pd = image_pd.sort_values(by=["id"])
    image_pd = image_pd.reset_index(drop=True)
    image_pd = image_pd.drop(["status_x", "status_y", "id"], axis=1)
    image_pd.columns = ["png_pre", "png_post"]
    image_pd["json_pre"] = image_pd["png_pre"].apply(
        lambda x: data_dir + "/labels/" + x.replace("png", "json"))
    image_pd["json_post"] = image_pd["png_post"].apply(
        lambda x: data_dir + "/labels/" + x.replace("png", "json"))
    image_pd["png_pre"] = image_pd["png_pre"].apply(
        lambda x: data_dir + "/images/" + x)
    image_pd["png_post"] = image_pd["png_post"].apply(
        lambda x: data_dir + "/images/" + x)
    image_pd.to_csv(data_dir+"/info.csv", index=False)


def save_weights(epoch: int, net: nn.Module, optimizer: op.Optimizer, save_path: str, model_name: str) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
    }
    torch.save(checkpoint, os.path.join(
        save_path, 'current_%s.pth' % (model_name)))
    if epoch % 4 == 0:
        torch.save(checkpoint, os.path.join(
            save_path, '%d_%s.pth' % (epoch, model_name)))


def weights_init(m:nn.Module):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def post_process(input:torch.Tensor, type:str, threshold:float=0.5) ->torch.Tensor:
    ones = torch.ones_like(input, device = input.device)
    zeros = torch.zeros_like(input, device = input.device)
    res = torch.where(input > threshold, ones, zeros)
    if type == "diff":
        res = onehot_to_number(res)
    return res

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1, 1, 1), device = real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.ones_like(d_interpolates, device = real_samples.device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty