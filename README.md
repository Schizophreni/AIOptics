# AIOptics
Physical system meets deep learning

## Training commands
**Stage I: emulating the physical system**
```
python main_stage1.py --data_path ../../Optics/ --resize_size 256 --save_path checkpoints/with_fft 
```
**Parameters**
- `data_path`: dataset folder, which contains a `train` and a `val` folder. Each folder contains four types of images / patterns `[object, edge, number, noise]`.
- `resize_size`: resize the raw image into smaller images for efficient training.
- `save_path`: path for saving checkpoints and logs.

**Notes**

>- For the pattern image, which may be the intensity of diffraction, we do not directly learn mapping from images to intensities. (Because most of the intensitie are focused at the center of images, given the pixels around near to `0`). Instead, we try to learn `log(I+e)`, where `e>0` is a small constant.
