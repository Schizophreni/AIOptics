import torch
import torch.nn.functional as F

def fft2d(x):
    """Applies 2D FFT and shifts zero frequency to the center."""
    x = x.to(torch.float32)
    return torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))

def compute_amp_phase(fft_complex):
    """Returns amplitude and phase tensors."""
    amplitude = torch.abs(fft_complex)
    phase = torch.angle(fft_complex)
    return amplitude, phase

def fourier_amp_phase_loss(pred, target, amp_weight=1.0, phase_weight=1.0):
    """
    Computes frequency-domain loss using both amplitude and phase.
    Args:
        pred: (B, 1, H, W) predicted image
        target: (B, 1, H, W) ground truth image
        amp_weight: weighting for amplitude loss
        phase_weight: weighting for phase loss
    Returns:
        scalar loss
    """
    # with torch.autocast(device_type="cuda", dtype=torch.float32):
    pred_fft = fft2d(pred.squeeze(1))
    target_fft = fft2d(target.squeeze(1))

    amp_pred, phase_pred = compute_amp_phase(pred_fft)
    amp_target, phase_target = compute_amp_phase(target_fft)

    amp_loss = F.mse_loss(amp_pred, amp_target)
    
    # Phase is angular → need to use wrap-around safe difference
    phase_diff = torch.sin((phase_pred - phase_target) / 2.0) ** 2
    phase_loss = phase_diff.mean()

    return amp_weight * amp_loss + phase_weight * phase_loss