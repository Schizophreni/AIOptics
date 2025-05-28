import torch
from model import img2pattern_model
import argparse
from datasets import PairedDataset
import os
import torchvision

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="path to the data")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="path to save the model")
    parser.add_argument("--pretrain", type=str, default="", help="path to load the pretrain model")
    parser.add_argument("--pattern_types", nargs="+", default=["object", "edge", "number", "noise"],)
    args = parser.parse_args()
    return args


args = get_args()
# load model
model = img2pattern_model(inp_channels=1, out_channels=1, dim=48)
model = model.cuda()
model.load_state_dict(torch.load(args.pretrain))

# construct dataloader
val_dataset = PairedDataset(data_dir=os.path.join(args.data_path, "val"), pattern_types=args.pattern_types, img_size=800)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# begin evaluation
os.makedirs(args.save_path, exist_ok=True)
model.eval()
for idx, (inp, _, inp_name) in enumerate(val_loader):
    print("processing image: ", inp_name[0])
    inp = inp.cuda()
    out_pred = model(inp).detach().cpu()
    out_intensity = torch.exp(out_pred).clamp(min=0, max=1.0)
    torchvision.utils.save_image(out_intensity, os.path.join(args.save_path, inp_name[0]))
