"""
Stage1: mimicking the physical system of transforming the image to optical pattern
Use torch mixed precision training to accelerate the training process
Ref: https://www.cnblogs.com/jimchen1218/p/14315008.html
"""
import os
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from model import img2pattern_model, pattern2img_model
from datasets import PairedDataset
from utils import psnr, ssim  # metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchvision
from layers import fourier_amp_phase_loss

pixel_criterion = torch.nn.L1Loss()  # pixel criterion

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="path to the data")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--gpu_ids", type=str, default="0", help="device")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="path to save the model")
    parser.add_argument("--resume_path", type=str, default="", help="path to load the model")
    parser.add_argument("--print_interval", type=int, default=10, help="print interval")
    parser.add_argument("--resize_size", type=int, default=256, help="resize size for images and patterns")
    parser.add_argument("--pattern_types", nargs="+", default=["object", "edge", "number", "noise"],)
    args = parser.parse_args()
    return args

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, scaler,
                    device, epoch, print_interval=10, writer=None, log_file=None):
    model.train()
    running_loss = 0.0
    running_con_loss = 0.0
    for i, (inp, tgt, mask) in enumerate(dataloader):
        inp, tgt = inp.to(device), tgt.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out_pred = model(inp)
            loss = (torch.abs(out_pred - log_intensity(tgt))).mean()
            con_loss = fourier_amp_phase_loss(normalize(out_pred), normalize(log_intensity(tgt)))
        scaler.scale(loss + 1e-1*con_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        running_loss += loss.item()
        running_con_loss += con_loss.item()
        # running_con_loss += loss.item() * 0.0
        if i % print_interval == 0:
            print(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.3f}, con_loss: {running_con_loss / print_interval:.3f}")
            writer.add_scalar("train loss", running_loss / print_interval, epoch * len(dataloader) + i)
            writer.add_scalar("train con loss", running_con_loss / print_interval, epoch * len(dataloader) + i)
            with open(log_file, "a") as f:
                f.write(f"[epoch: {epoch + 1}, iter: {i + 1:5d}] loss: {running_loss / print_interval:.3f}\n, con_loss: {running_con_loss / print_interval:.3f}")
            running_loss = 0.0
            running_con_loss = 0.0
    lr_scheduler.step()

def validate(model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for idx, (inp, out, _) in tqdm(enumerate(val_loader), ncols=60, desc="Validating"):
            inp, out = inp.to(device), out.to(device)
            out_pred = model(inp)
            out = normalize(log_intensity(out))
            out_pred = normalize(out_pred)
            if idx < 10:
                torchvision.utils.save_image(inp.cpu(), os.path.join("preds", "img_{}.png".format(idx+1)))
                torchvision.utils.save_image(out.cpu(), os.path.join("preds", "pattern_{}.png".format(idx+1)))
                torchvision.utils.save_image(out_pred.cpu(), os.path.join("preds", "pred_{}.png".format(idx+1)))
            psnrs.append(psnr(out_pred*255, out*255).item())
            ssims.append(ssim(out_pred*255, out*255).item())
    psnr_mean = np.mean(np.array(psnrs))
    ssim_mean = np.mean(np.array(ssims))
    return psnr_mean, ssim_mean

def log_intensity(intensity, eps=1e-6):
    return torch.log(intensity + 1e-6)

def normalize(inp, min_val=0.0, max_val=1.0):
    inp = (inp - inp.min().detach()) / (inp.max().detach() - inp.min().detach() + 1e-6)
    inp = min_val + (max_val - min_val) * inp
    return inp

def load_checkpoint(model, optimizer, lr_scheduler, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    best_psnr = 18 #checkpoint["best_psnr"]
    return checkpoint["epoch"], best_psnr

def main(args):
    # set devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda") if len(args.gpu_ids) > 0 else torch.device("cpu")
    save_path = os.path.join(args.save_path, "stage1")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = os.path.join(save_path, "log.txt")
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    # build model
    model = img2pattern_model(inp_channels=1, out_channels=1, dim=48, num_blocks=[4, 6, 6, 8])
    model = model.to(device)
    # build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-8)
    scaler = GradScaler()
    # resume from checkpoint
    if os.path.exists(args.resume_path):
        start_epoch, best_psnr = load_checkpoint(model, optimizer, lr_scheduler, scaler, args.resume_path)
    else:
        start_epoch, best_psnr = 0, 0.0
    # build dataloader
    train_dataset = PairedDataset(data_dir=os.path.join(args.data_path, "train"), pattern_types=args.pattern_types, img_size=args.resize_size)
    val_dataset = PairedDataset(data_dir=os.path.join(args.data_path, "val"), pattern_types=args.pattern_types, img_size=args.resize_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # train
    for epoch in range(start_epoch, args.num_epochs):
        train_one_epoch(model, train_loader, optimizer, lr_scheduler, scaler, device, epoch, print_interval=args.print_interval,
                        writer=writer, log_file=log_file)
        # save checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr
        }, os.path.join(save_path, "latest.pth"))
        # validate
        psnr_mean, ssim_mean = validate(model, val_loader, device)
        writer.add_scalar("val PSNR", psnr_mean, epoch + 1)
        writer.add_scalar("val SSIM", ssim_mean, epoch + 1)
        print(f"[{epoch + 1}] val PSNR: {psnr_mean:.3f}, val SSIM: {ssim_mean:.3f}\n")
        with open(log_file, "a") as f:
            f.write(f"[{epoch + 1}] val PSNR: {psnr_mean:.3f}, val SSIM: {ssim_mean:.3f}\n")
        if psnr_mean > best_psnr:
            best_psnr = psnr_mean
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
    writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)