import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
import os
from pathlib import Path
from PIL import Image


class PairedDataset(Data.Dataset):
    """
    Paired dataset for correspond input / output image pairs
    """
    def __init__(self, data_dir, pattern_types:list, img_size:int) -> None:
        super().__init__()
        """
        data_dir: root folder of images
        pattern_types: list | 1-object 2-edge 3-numeber 4-noise
        img_size: cropped img size
        """
        self.data_dir = data_dir
        self.pattern_types = pattern_types
        self.img_size = img_size
        self.obtain_imgs()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=img_size)
    
    def parse_subfolder(self, pattern_type):
        """
        obtain input / output images within a subfolder, e.g., 1-object
        """
        print(f"==> Parsing pattern type: {pattern_type}")
        inputs, outputs = [], []
        input_path = Path(os.path.join(self.data_dir, pattern_type, "images"))
        output_path = Path(os.path.join(self.data_dir, pattern_type, "patterns"))
        imgs = os.listdir(input_path)
        for img in imgs:
            if ".png" in img or ".jpg" in img:
                inputs.append(input_path / img)
                outputs.append(output_path / img)
        return inputs, outputs
    
    def obtain_imgs(self):
        self.input_imgs, self.output_imgs = [], []
        for p_type in self.pattern_types:
            inputs, outputs = self.parse_subfolder(pattern_type=p_type)
            self.input_imgs.extend(inputs)
            self.output_imgs.extend(outputs)
    
    def __getitem__(self, index):
        inp, out = self.input_imgs[index], self.output_imgs[index]
        inp, out = Image.open(inp), Image.open(out)
        inp, out = self.to_tensor(inp), self.to_tensor(out)
        inp, out = self.resize(inp), self.resize(out)
        return inp, out, self.input_imgs[index].split("/")[-1].split("\\")[-1]
    
    def __len__(self):
        return len(self.input_imgs)
    
    def check_effect_resolution(self):
        """
        input / output images are of resolution 800x800, with 0 around the border
        we use center crop to obtain effect resolution by excluding 0 
        """
        for i in range(self.__len__()):
            inp, out = self.__getitem__(i)
            inp, out = (255*inp).to(torch.int32), (255*out).to(torch.int32)
            crop_inp, crop_out = self.center_crop(inp), self.center_crop(out)
            diff_inp = inp.sum() - crop_inp.sum()
            diff_out = out.sum() - crop_out.sum()
            if diff_inp > 0 or diff_out > 0:
                print(i, self.input_imgs[i], diff_inp, diff_out)
                print(f"Resolution {self.img_size}x{self.img_size} check failded, try a larger resolution")
                break
            print("Resolution check passed !")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = PairedDataset(data_dir="/Users/wuran/Downloads/Optics/val", pattern_types=["object"], img_size=512)
    # dataset.check_effect_resolution()
    print(len(dataset))
    for i in range(len(dataset)):
        inp, out = dataset[i]
        out = torch.log(out + 1e-6)
        out = (out - out.min()) / (out.max() - out.min())
        plt.subplot(1, 2, 1)
        plt.imshow(inp.permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(out.permute(1, 2, 0))
        plt.show()
        






    
