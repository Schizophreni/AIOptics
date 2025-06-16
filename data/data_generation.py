import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm


def to_complex(a):
    return a.to(torch.complex64)

def H(dim, lmb, d, pixel_size):
    df = 1 / (pixel_size * dim)
    fx = torch.linspace(-dim / 2, dim / 2 - 1, dim) * df * lmb
    fy = torch.linspace(-dim / 2, dim / 2 - 1, dim) * df * lmb
    fY, fX = torch.meshgrid(fy, fx, indexing='ij')
    fR = fX ** 2 + fY ** 2
    condition = (2 * np.pi / lmb) ** 2 * (1 - fR)
    tmp = to_complex(torch.sqrt(torch.abs(condition)))
    kz = torch.where(condition >= 0, tmp, 1j * tmp)
    return torch.exp(1j * kz * d)

def a0_to_uz(a0, funcH):
    az = a0 * torch.fft.fftshift(funcH)
    uz = torch.fft.ifft2(az)
    return uz

def diff(input_, funcH, dim_out=None):
    device = input_.device
    dim_in = input_.shape[-1]
    dim_all = funcH.shape[-1]
    border_in = (dim_all - dim_in) // 2
    dim_out = dim_out or dim_in
    border_out = (dim_all - dim_out) // 2

    u0 = F.pad(input_, (border_in, border_in, border_in, border_in), mode='constant', value=0).to(torch.complex64)
    a0 = torch.fft.fft2(u0)
    uz = a0_to_uz(a0, funcH.to(device))

    if border_out > 0:
        return uz[..., border_out:-border_out, border_out:-border_out]
    return uz

def norm(_input):
    max_i = np.max(_input)
    min_i = np.min(_input)
    return (_input - min_i) / (max_i - min_i)

def pad_image(image, target_height, target_width):
    h, w = image.shape
    if target_height < h:
        crop_top = (h - target_height) // 2
        crop_bottom = crop_top + target_height
        image = image[crop_top:crop_bottom, :]

    if target_width < w:
        crop_left = (w - target_width) // 2
        crop_right = crop_left + target_width
        image = image[:, crop_left:crop_right]

    h, w = image.shape
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    # return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

def full_propagation(image, H1, phase, amplitude):
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if amplitude.ndim == 2:
        amplitude = amplitude.unsqueeze(0)

    image = image.to(torch.complex64)
    phase = phase.to(torch.complex64)
    amplitude = amplitude.to(torch.complex64)

    p1 = diff(image, H1)
    p1_phased = p1 * torch.exp(-1j * phase)
    p2 = torch.abs(diff(p1_phased, H1)) ** 2
    return p2.squeeze()

if __name__ == "__main__":
    N, w = 256, 102
    pixel_size, wavelength = 8e-6, 594e-9
    d = 3e-3
    times = 32
    phase0 = (2 * np.pi * np.load("256repeatto1024phase.npy")).astype(np.float32)
    phase_tensor = torch.tensor(phase0).view(1, 1, 1024, 1024)
    phase_avg = F.avg_pool2d(phase_tensor, kernel_size=4, stride=4)[0, 0]
    H1 = H(times*N, wavelength, d, pixel_size)
    amplitude = torch.ones((N, N))
    for i in tqdm(range(4500), desc="generate train"):
        image = torch.rand(w, w)*255.0
        pad_img = pad_image(image, target_height=N, target_width=N)
        intensity = full_propagation(pad_img, H1, phase_avg, amplitude=amplitude) / 1000
        image_file = "train/images/{}.npy".format(i+1)
        pat_file = "train/patterns/{}.npy".format(i+1)
        np.save(image_file, image.numpy().astype("uint8"))
        np.save(pat_file, intensity.numpy().astype("f4"))
    for i in tqdm(range(500), desc="generate val"):
        image = torch.rand(w, w)*255.0
        pad_img = pad_image(image, target_height=N, target_width=N)
        intensity = full_propagation(pad_img, H1, phase_avg, amplitude=amplitude) / 1000
        image_file = "val/images/{}.npy".format(i+4501)
        pat_file = "val/patterns/{}.npy".format(i+4501)
        np.save(image_file, image.numpy().astype("uint8"))
        np.save(pat_file, intensity.numpy().astype("f4"))