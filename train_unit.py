import torch
import torch.nn as nn
import torch.optim as op
from typing import Optional
from torch import autocast
from torch.cuda.amp import GradScaler
from utils import compute_gradient_penalty,post_process


def update_locD(net_locD: nn.Module, net_locG: nn.Module, optim_locD: op.Optimizer,
                img_pre: torch.Tensor, label_pre: torch.Tensor, scaler: Optional[GradScaler]= None,
                loss: str = "wgan", clip:Optional[float] = None, use_origin: bool = True) -> torch.Tensor:
    if loss == "wgan":
        assert clip is not None, "wgan loss need a clip value"
    real_label = torch.ones(
        size=(img_pre.shape[0],), dtype=torch.float32, device=img_pre.device)
    fake_label = torch.zeros(
        size=(img_pre.shape[0],), dtype=torch.float32, device=img_pre.device)

    net_locD.zero_grad()
    if scaler is not None:
        with autocast(device_type='cuda', dtype=torch.float16):
            fake_loc = net_locG(img_pre).detach().float()
            if use_origin:
                real_loc = torch.cat((img_pre, label_pre), dim=1)
                fake_loc = torch.cat((img_pre, fake_loc), dim=1)
            else:
                real_loc = label_pre.float()
            
            locD_real = net_locD(real_loc)
            locD_fake = net_locD(fake_loc)
            
            if loss == "gan":
                loss_func = nn.BCEWithLogitsLoss()
                err_locD_real = loss_func(locD_real, real_label)
                err_locD_fake = loss_func(locD_fake, fake_label)
                err_locD = err_locD_real * 0.5 + err_locD_fake * 0.5
            elif loss == "wgan":
                err_locD_real = torch.mean(locD_real)
                err_locD_fake = torch.mean(locD_fake)
                err_locD = -err_locD_real + err_locD_fake
        scaler.scale(err_locD).backward()
        scaler.step(optim_locD)
        if loss == "wgan":
            scaler.unscale_(optim_locD)
            for p in net_locD.parameters():
                p.data.clamp_(-clip, clip)
    else:
        fake_loc = net_locG(img_pre).detach().float()
        fake_loc = post_process(fake_loc, "loc")
        if use_origin:
            real_loc = torch.cat((img_pre, label_pre), dim=1)
            fake_loc = torch.cat((img_pre, fake_loc), dim=1)
        else:
            real_loc = label_pre.float()
        locD_real = net_locD(real_loc)
        locD_fake = net_locD(fake_loc)
        
        if loss == "gan":
            loss_func = nn.BCEWithLogitsLoss()
            err_locD_real = loss_func(locD_real, real_label)
            err_locD_fake = loss_func(locD_fake, fake_label)
            err_locD = err_locD_real * 0.5 + err_locD_fake * 0.5
        elif loss == "wgan":
            err_locD_real = torch.mean(locD_real)
            err_locD_fake = torch.mean(locD_fake)
            err_locD = -err_locD_real + err_locD_fake
        elif loss== "wgan-gp":
            err_locD_real = torch.mean(locD_real)
            err_locD_fake = torch.mean(locD_fake)
            gp = compute_gradient_penalty(net_locD, real_loc, fake_loc)
            err_locD = -err_locD_real + err_locD_fake + 10 * gp
        err_locD.backward()
        optim_locD.step()
        if loss == "wgan":
            for p in net_locD.parameters():
                p.data.clamp_(-clip, clip)

    return err_locD


def update_locG(net_locD: nn.Module, net_locG: nn.Module, optim_locG: op.Optimizer,
                img_pre: torch.Tensor, label_pre: torch.Tensor, scaler: Optional[GradScaler]= None,
                loss: str = "wgan", clip:Optional[float] = None, use_origin: bool = True) -> torch.Tensor:
    if loss == "wgan":
        assert clip is not None, "wgan loss need a clip value"
    fake_label = torch.zeros(
        size=(img_pre.shape[0],), dtype=torch.float32, device=img_pre.device)
    l1loss = nn.L1Loss()

    net_locG.zero_grad()
    if scaler is not None:
        with autocast(device_type='cuda', dtype=torch.float16):
            fake_loc = net_locG(img_pre).float()
            if use_origin:
                fake_loc = torch.cat((img_pre, fake_loc), dim=1)
                
            locD_fake = net_locD(fake_loc)
            
            if loss == "gan":
                loss_func = nn.BCEWithLogitsLoss()
                err_locD_fake = loss_func(locD_fake, fake_label)
                err_locG_l1 = l1loss(fake_loc, label_pre)
                err_locG = err_locG_l1 * 0.5 + err_locD_fake * 0.5
            elif loss=="wgan" or loss == "wgan-gp":
                err_locG = -torch.mean(locD_fake)

        scaler.scale(err_locG).backward()
        scaler.step(optim_locG)
    else:
        fake_loc = net_locG(img_pre).float()
        if use_origin:
            fake_loc_x = torch.cat((img_pre, fake_loc), dim=1)
        else:
            fake_loc_x = fake_loc.float()
        
        locD_fake = net_locD(fake_loc_x)
        
        if loss == "gan":
            loss_func = nn.BCEWithLogitsLoss()
            err_locD_fake = loss_func(locD_fake, fake_label)
            err_locG_l1 = l1loss(fake_loc, label_pre)
            err_locG = err_locG_l1 * 0.5 + err_locD_fake * 0.5
        elif loss == "wgan" or loss == "wgan-gp":
            err_locG = -torch.mean(locD_fake)
        err_locG.backward()
    return err_locG, fake_loc


def update_diffD(net_locG: nn.Module, net_diffD: nn.Module, net_diffG: nn.Module, optim_diffD: op.Optimizer,
                 img_pre: torch.Tensor, img_post: torch.Tensor, label_post: torch.Tensor, scaler: Optional[GradScaler]= None,
                loss: str = "wgan", clip:Optional[float] = None, use_origin: bool = True) -> torch.Tensor:
    if loss == "wgan":
        assert clip is not None, "wgan loss need a clip value"
    real = torch.ones(
        size=(img_pre.shape[0],), dtype=torch.float32, device=img_pre.device)
    fake_label = torch.zeros(
        size=(img_pre.shape[0],), dtype=torch.float32, device=img_pre.device)

    net_diffD.zero_grad()
    if scaler is not None:
        with autocast(device_type='cuda', dtype=torch.float16):
            fake_loc = torch.cat(
                (img_pre, img_post, net_locG(img_pre).detach()), dim=1)
            fake_diff = net_diffG(fake_loc).detach().float()
            if use_origin:
                real_diff = torch.cat((img_pre, img_post, label_post), dim=1)
                fake_diff = torch.cat((img_pre, img_post, fake_diff), dim=1)
            else:
                read_diff = label_post.float()
            
            diffD_real = net_diffD(real_diff)
            diffD_fake = net_diffD(fake_diff)
            
            if loss == "gan":
                loss_func = nn.BCEWithLogitsLoss()
                err_diffD_real = loss_func(diffD_real, real_label)
                err_diffD_fake = loss_func(diffD_fake, fake_label)
                err_diffD = err_diffD_real * 0.5 + err_diffD_fake * 0.5
            elif loss == "wgan":
                err_diffD_real = torch.mean(diffD_real)
                err_diffD_fake = torch.mean(diffD_fake)
                err_diffD = -err_diffD_real + err_diffD_fake
            
        scaler.scale(err_diffD).backward()
        scaler.step(optim_diffD)
        if loss == "wgan":
            scaler.unscale_(optim_diffD)
            for p in net_diffD.parameters():
                p.data.clamp_(-clip, clip)
    else:
        fake_loc = torch.cat(
            (img_pre, img_post, net_locG(img_pre).detach()), dim=1)
        fake_diff = net_diffG(fake_loc).detach().float()
        if use_origin:
            real_diff = torch.cat((img_pre, img_post, label_post), dim=1)
            fake_diff = torch.cat((img_pre, img_post, fake_diff), dim=1)
        else:
            real_diff = label_post.float()

        diffD_real = net_diffD(real_diff)
        diffD_fake = net_diffD(fake_diff)
        
        if loss == "gan":
            loss_func = nn.BCEWithLogitsLoss()
            err_diffD_real = loss_func(diffD_real, real_label)
            err_diffD_fake = loss_func(diffD_fake, fake_label)
            err_diffD = err_diffD_real * 0.5 + err_diffD_fake * 0.5
        elif loss == "wgan":
            err_diffD_real = torch.mean(diffD_real)
            err_diffD_fake = torch.mean(diffD_fake)
            err_diffD = -err_diffD_real + err_diffD_fake
        elif loss == "wgan-gp":
            err_diffD_real = torch.mean(diffD_real)
            err_diffD_fake = torch.mean(diffD_fake)
            gp = compute_gradient_penalty(net_diffD, real_diff, fake_diff)
            err_diffD = -err_diffD_real + err_diffD_fake + 10 * gp
        err_diffD.backward()
        optim_diffD.step()
        if loss == "wgan":
            for p in net_diffD.parameters():
                p.data.clamp_(-clip, clip)

    return err_diffD


def update_diffG(net_locG: nn.Module, net_diffD: nn.Module, net_diffG: nn.Module, optim_diffG: op.Optimizer,
                 img_pre: torch.Tensor, img_post: torch.Tensor, label_post: torch.Tensor, scaler: Optional[GradScaler]= None,
                 loss: str = "wgan", clip:Optional[float] = None, use_origin: bool = True) -> torch.Tensor:
    if loss == "wgan":
        assert clip is not None, "wgan loss need a clip value"
    fake_label = torch.zeros(
        size=(img_pre.shape[0],), dtype=torch.float32, device=img_pre.device)
    l1loss = nn.L1Loss()

    net_diffG.zero_grad()
    if scaler is not None:
        with autocast(device_type='cuda', dtype=torch.float16):
            fake_loc = torch.cat(
                (img_pre, img_post, net_locG(img_pre).detach()), dim=1)
            fake_diff = net_diffG(fake_loc).float()
            if use_origin:
                fake_diff = torch.cat((img_pre, img_post, fake_diff), dim=1)
            diffD_fake = net_diffD(fake_diff)
            
            if loss == "gan":
                loss_func = nn.BCEWithLogitsLoss()
                err_diffD_fake = loss_func(diffD_fake, fake_label)
                err_diffG_l1 = l1loss(fake_diff, label_post)
                err_diffG = err_diffG_l1 * 0.5 + err_diffD_fake * 0.5
            elif loss == "wgan" or loss == "wgan-gp":
                err_diffG = -torch.mean(diffD_fake)
        scaler.scale(err_diffG).backward()
        scaler.step(optim_diffG)
    else:
        fake_loc = torch.cat(
                    (img_pre, img_post, net_locG(img_pre).detach()), dim=1)
        fake_diff = net_diffG(fake_loc).float()
        if use_origin:
            fake_diff = torch.cat((img_pre, img_post, fake_diff), dim=1)
        diffD_fake = net_diffD(fake_diff)
        
        if loss == "gan":
            loss_func = nn.BCEWithLogitsLoss()
            err_diffD_fake = loss_func(diffD_fake, fake_label)
            err_diffG_l1 = l1loss(fake_diff, label_post)
            err_diffG = err_diffG_l1 * 0.5 + err_diffD_fake * 0.5
        elif loss == "wgan" or loss == "wgan-gp":
            err_diffG = -torch.mean(diffD_fake)
        err_diffG.backward()
        optim_diffG.step()

    return err_diffG, fake_diff