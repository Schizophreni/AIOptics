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
