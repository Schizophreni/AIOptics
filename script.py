import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data.data_generation import *
from PIL import Image
import cv2


# img = "/Users/wuran/Downloads/OpticsData/train/object/images/1.png"
# image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
# image = image.astype(np.float32) / 255.0
# image_tensor = torch.tensor(image).to(torch.float32)

phase0 = (2 * np.pi * np.load("data/256repeatto1024phase.npy")).astype(np.float32)
phase_tensor = torch.tensor(phase0).view(1, 1, 1024, 1024)
phase_avg = F.avg_pool2d(phase_tensor, kernel_size=4, stride=4)[0, 0]

# img = torch.rand((102, 102)) * 255.0
# img = pad_image(img, 256, 256)
img = np.load("/Users/wuran/Downloads/1000.npy") # [256, 256]
img = torch.from_numpy(img).clamp(0, 255.0)
img = pad_image(img, 256, 256)
H1 = H(32*256, 594e-9, 3e-3, 8e-6)
intensity = full_propagation(img, H1, phase_avg, amplitude=torch.ones((256, 256))) / 1000
print("Max and min intensity: ", intensity.max(), intensity.min())

print((intensity.numpy() / 500 > 0.00001).sum())

plt.subplot(1, 3, 1)
plt.imshow(img.numpy())
plt.subplot(1, 3, 2)
plt.imshow(intensity.numpy() / 500, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(intensity.numpy() / 500 > 0.00001, cmap="gray")
plt.show()

