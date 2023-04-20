import torch
import torch.nn as nn
import torch.optim as op
from xview_data import XVIEW_DATA
from model import netD, netG, unetG
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from utils import save_weights, weights_init, onehot_to_number, post_process
from train_unit import update_locD, update_locG, update_diffD, update_diffG
from torchsummary import summary
from torch.profiler import profile, record_function, ProfilerActivity
import config as cf
from evaluate import Eva, Eva_loc
import pandas as pd
from pprint import pprint

device = cf.device
if not os.path.exists(cf.save_dir):
    os.mkdir(cf.save_dir)
os.system(f"cp -r ../code {cf.save_dir}")
log_file = open(f"{cf.save_dir}/train.log", "a+") if cf.output_to_file else None
tqdm.write(f"now work type is {device}", file = log_file)
tqdm.write(f"now gpu num is {torch.cuda.device_count()}", file = log_file)

train, val, out = random_split(XVIEW_DATA(cf.data_dir), [
                               cf.train_rate, cf.val_rate, 1 - cf.train_rate - cf.val_rate])
train_loader = DataLoader(
    train, shuffle=True, batch_size=cf.batch_size, num_workers=cf.num_workers, pin_memory=True)
val_loader = DataLoader(
    val, shuffle=False, batch_size=cf.batch_size, num_workers=cf.num_workers, pin_memory=True)

if cf.use_origin:
    # img_pre + label_pre(fake_label_pre) -> bool
    # (3, 1024, 1024) + (1, 1024, 1024) -> {0, 1}
    net_loc_D = netD(4, 64)
    # img_pre + img_post + label_post(fake_label_post) -> bool
    # (3, 1024, 1024) + (3, 1024, 1024) + (4, 1024, 1024) -> {0, 1}
    net_diff_D = netD(10, 64)
else:
    # label_pre(fake_label_pre) -> bool
    # (1, 1024, 1024) -> {0, 1}
    net_loc_D = netD(1, 64)
    # label_post(fake_label_post) -> bool
    # (4, 1024, 1024) -> {0, 1}
    net_diff_D = netD(4, 64)

# img_pre -> label_pre
# (3 ,1024, 1024) -> (1, 1024, 1024)
if cf.use_unet_loc:
    net_loc_G = unetG(3, 1)
else:
    net_loc_G = netG(3, 8, 1)
    

# img_pre + img_post + label_pre -> label_post
# (3, 1024, 1024) + (3, 1024, 1024) + (1, 1024, 1024) -> (4, 1024, 1024)
if cf.use_unet_diff:
    net_diff_G = unetG(7, 4)
else:
    net_diff_G = netG(7, 8, 4)
    

net_loc_D.apply(weights_init)
net_diff_D.apply(weights_init)
net_loc_G.apply(weights_init)
net_diff_G.apply(weights_init)


if cf.parallel:
    nn.DataParallel(net_loc_D).to(device)
    nn.DataParallel(net_loc_G).to(device)
    nn.DataParallel(net_diff_D).to(device)
    nn.DataParallel(net_diff_G).to(device)
else:
    net_loc_D.to(device)
    net_loc_G.to(device)
    net_diff_D.to(device)
    net_diff_G.to(device)

if cf.summary_model:
    if cf.use_origin:
        summary(net_loc_D, (4, 1024, 1024))
        summary(net_loc_G, (3, 1024, 1024))
        summary(net_diff_D, (10, 1024, 1024))
        summary(net_diff_G, (7, 1024, 1024))
    else:
        summary(net_loc_D, (1, 1024, 1024))
        summary(net_loc_G, (3, 1024, 1024))
        summary(net_diff_D, (4, 1024, 1024))
        summary(net_diff_G, (7, 1024, 1024))

if cf.RESUME:
    if not cf.loc_only:
        assert os.path.exists(os.path.join(cf.save_dir, 'current_net_loc_G.pth')) \
            and os.path.exists(os.path.join(cf.save_dir, 'current_net_loc_D.pth')) \
            and os.path.exists(os.path.join(cf.save_dir, 'current_net_diff_G.pth')) \
            and os.path.exists(os.path.join(cf.save_dir, 'current_net_diff_D.pth')), \
            'There is not found any saved weights'
    else:
        assert os.path.exists(os.path.join(cf.save_dir, 'current_net_loc_G.pth')) \
        and os.path.exists(os.path.join(cf.save_dir, 'current_net_loc_D.pth')), \
        'There is not found any saved weights'
    tqdm.write("\nLoading pre-trained networks.", file = log_file)
    init_epoch = torch.load(os.path.join(
        cf.save_dir, 'current_net_loc_D.pth'))['epoch']
    net_loc_D.load_state_dict(torch.load(os.path.join(
        cf.save_dir, 'current_net_loc_D.pth'))['model_state_dict'])
    net_loc_G.load_state_dict(torch.load(os.path.join(
        cf.save_dir, 'current_net_loc_G.pth'))['model_state_dict'])
    if not cf.loc_only:
        net_diff_D.load_state_dict(torch.load(os.path.join(
            cf.save_dir, 'current_net_diff_D.pth'))['model_state_dict'])
        net_diff_G.load_state_dict(torch.load(os.path.join(
            cf.save_dir, 'current_net_diff_G.pth'))['model_state_dict'])
    tqdm.write("Done.\n", file = log_file)


loc_D = op.Adam(net_loc_D.parameters(), lr=cf.d_lr, betas = cf.beta)
loc_G = op.Adam(net_loc_G.parameters(), lr=cf.g_lr, betas = cf.beta)
diff_D = op.Adam(net_diff_D.parameters(), lr=cf.d_lr, betas = cf.beta)
diff_G = op.Adam(net_diff_G.parameters(), lr=cf.g_lr, betas = cf.beta)
scaler = torch.cuda.amp.GradScaler() if cf.half_precision else None


bar_format = '{desc}{percentage:3.0f}%|{bar}|{remaining}[{n_fmt}/{total_fmt}{postfix}]'



for epoch in range(cf.num_epoches):
    # prof = torch.profiler.profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('/root/tf-logs/net'),
    #     record_shapes=True,
    #     profile_memory=True)
    # prof.start()
    with tqdm(train_loader, ncols=90, bar_format=bar_format, file = log_file, desc="training:") as t:
        for i, data in enumerate(t, 0):
            img_pre, img_post, label_pre, label_post = [
                img.to(device=device) for img in data]
            for _ in range(cf.d_per_g):
                err_locD = update_locD(net_loc_D, net_loc_G, loc_D,
                                       img_pre, label_pre, scaler=scaler,
                                       loss=cf.loss, clip = cf.clip, use_origin = cf.use_origin)
                if not cf.loc_only:
                    err_diffD = update_diffD(net_loc_G, net_diff_D, net_diff_G, diff_D,
                                             img_pre, img_post, label_post, scaler=scaler,
                                             loss=cf.loss, clip = cf.clip, use_origin = cf.use_origin)
                if scaler is not None:
                    scaler.update()
            for _ in range(cf.g_per_d):
                err_locG, fake_loc = update_locG(net_loc_D, net_loc_G, loc_G,
                                       img_pre, label_pre, scaler=scaler,
                                       loss=cf.loss, clip = cf.clip, use_origin = cf.use_origin)
                if not cf.loc_only:
                    err_diffG, fake_diff = update_diffG(net_loc_G, net_diff_D, net_diff_G, diff_G,
                                             img_pre, img_post, label_post, scaler=scaler,
                                             loss=cf.loss, clip = cf.clip, use_origin = cf.use_origin)
                if scaler is not None:
                    scaler.update()

            if not cf.loc_only:
                t.set_postfix({"locD": "{:.2f}".format(err_locD.item()),
                               "locG:": "{:.2f}".format(err_locG.item()),
                               "diffD": "{:.2f}".format(err_diffD.item()),
                               "diffG": "{:.2f}".format(err_diffG.item())})
            else:
                t.set_postfix({"locD": "{:.2f}".format(err_locD.item()),
                               "locG": "{:.2f}".format(err_locG.item())})
            if cf.BOARD:
                from torch.utils.tensorboard import SummaryWriter
                fake_loc = post_process(fake_loc, "loc")
                label_loc = label_pre*255
                fake_loc = fake_loc*255
                if not cf.loc_only:
                    fake_diff = post_process(fake_diff, "diff")
                    fake_diff =  (onehot_to_number(fake_diff)*255/4).floor()
                    label_diff = (onehot_to_number(label_post)*255/4).floor()
                
                writer = SummaryWriter(cf.tensorboard_log_dir+"/train")
                writer.add_images("image_pre", img_pre, i)
                writer.add_images("image_post", img_post, i)
                writer.add_images("label_loc", label_loc, i)
                writer.add_images("predict_loc", fake_loc, i)
                if not cf.loc_only:
                    writer.add_images("label_diff", label_diff, i)
                    writer.add_images("predict_diff", fake_diff, i)
                
                
                
                for tag, value in net_loc_D.named_parameters():
                    tag = "locD/" + tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), i)
                    writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i)
                for tag, value in net_loc_G.named_parameters():
                    tag = "locG/" + tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), i)
                    writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i)
                if not cf.loc_only:
                    for tag, value in net_diff_D.named_parameters():
                        tag = "diffD/" + tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), i)
                        writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i)
                    for tag, value in net_diff_G.named_parameters():
                        tag = "diffG/" + tag.replace('.', '/')
                        writer.add_histogram(tag, value.data.cpu().numpy(), i)
                        writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), i)

    #         prof.step()
    # prof.stop()

    tqdm.write(f"epoch:{epoch} ending,saving...", file = log_file)
    save_weights(epoch, net_loc_D, loc_D, cf.save_dir, "net_loc_D")
    save_weights(epoch, net_loc_G, loc_G, cf.save_dir, "net_loc_G")
    if not cf.loc_only:
        save_weights(epoch, net_diff_D, diff_D, cf.save_dir, "net_diff_D")
        save_weights(epoch, net_diff_G, diff_G, cf.save_dir, "net_diff_G")
        
    if cf.VALIDATE:
        zero = torch.zeros((1, 1, 1024, 1024), device=device)
        # init a Eva, with FP=0, FN=0, TP=0
        if not cf.loc_only:
            score = Eva(zero, zero, zero, zero)
        else:
            score = Eva_loc(zero, zero)
        with net_diff_G.eval() and net_loc_G.eval() and torch.no_grad():
            with tqdm(val_loader, ncols=80, bar_format=bar_format, file = log_file, desc="validating:") as t:
                for i, data in enumerate(t, 0):
                    img_pre, img_post, label_pre, label_post = [
                        img.to(device=device) for img in data]
                    loc_pred = net_loc_G(img_pre)
                    loc_pred = post_process(loc_pred, "loc")

                    if not cf.loc_only:
                        fake_loc = torch.cat((img_pre, img_post, loc_pred), dim=1)
                        des_pred = net_diff_G(fake_loc)
                        des_pred = post_process(des_pred, "diff")
                        label_post = onehot_to_number(label_post)
                        temp = Eva(loc_pred, des_pred, label_pre, label_post)
                        score += temp
                        t.set_postfix({
                            "loc_f1": "{:.2f}".format(score.loc_f1()),
                            "damage_f1": "{:.2f}".format(score.damage_f1())
                        })
                    else:
                        temp = Eva_loc(loc_pred, label_pre)
                        score += temp
                        t.set_postfix({"loc_f1": "{:.2f}".format(score.f1())})
                if cf.BOARD:
                    from torch.utils.tensorboard import SummaryWriter
                    label_loc = label_pre*255
                    fake_loc = loc_pred*255
                    if not cf.loc_only:
                        fake_diff =  (onehot_to_number(des_pred)*255/4).floor()
                        label_diff = (onehot_to_number(label_post)*255/4).floor()

                    writer = SummaryWriter(cf.tensorboard_log_dir+"/validate")
                    writer.add_images("val_image_pre", img_pre, i)
                    writer.add_images("val_image_post", img_post, i)
                    writer.add_images("val_label_loc", label_loc, i)
                    writer.add_images("val_predict_loc", fake_loc, i)
                    if not cf.loc_only:
                        writer.add_images("val_label_diff", label_diff, i)
                        writer.add_images("val_predict_diff", fake_diff, i)



log_file.close()