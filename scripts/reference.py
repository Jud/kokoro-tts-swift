"""PyTorch reference implementations — DO NOT MODIFY.

These define the ground truth for CoreML fidelity comparison.
CustomSTFT replaces the original STFT with a conv1d-based version.
patch_sinegen_for_export makes the model deterministic for testing.

Imported by both export_coreml.py and stage_harness.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CustomSTFT — conv1d-based STFT replacement (no complex ops, CoreML-safe)
# ---------------------------------------------------------------------------

class CustomSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window="hann", center=True, pad_mode="replicate"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode
        self.freq_bins = self.n_fft // 2 + 1

        assert window == 'hann', window
        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if self.win_length < self.n_fft:
            window_tensor = F.pad(window_tensor, (0, self.n_fft - self.win_length))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[:self.n_fft]

        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)

        forward_window = window_tensor.numpy()
        forward_real = dft_real * forward_window
        forward_imag = dft_imag * forward_window

        self.register_buffer("weight_forward_fused",
                             torch.cat([
                                 torch.from_numpy(forward_real).float().unsqueeze(1),
                                 torch.from_numpy(forward_imag).float().unsqueeze(1),
                             ], dim=0))

        inv_scale = 1.0 / self.n_fft
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft
        idft_cos = np.cos(angle_t).T
        idft_sin = np.sin(angle_t).T
        inv_window = window_tensor.numpy() * inv_scale

        self.register_buffer("weight_backward_fused",
                             torch.cat([
                                 torch.from_numpy(idft_cos * inv_window).float().unsqueeze(1),
                                 -torch.from_numpy(idft_sin * inv_window).float().unsqueeze(1),
                             ], dim=0))

    def transform(self, waveform):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        ri = F.conv1d(x, self.weight_forward_fused, bias=None,
                      stride=self.hop_length, padding=0)
        real_out = ri[:, :self.freq_bins, :]
        imag_out = ri[:, self.freq_bins:, :]
        magnitude = torch.sqrt(real_out * real_out + imag_out * imag_out + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase

    def inverse(self, magnitude, phase, length=None):
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        combined = torch.cat([real_part, imag_part], dim=1)
        waveform = F.conv_transpose1d(combined, self.weight_backward_fused,
                                      bias=None, stride=self.hop_length, padding=0)
        if self.center:
            pad_len = self.n_fft // 2
            waveform = waveform[..., pad_len:-pad_len]
        if length is not None:
            waveform = waveform[..., :length]
        return waveform

    def forward(self, x):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])


# ---------------------------------------------------------------------------
# SineGen — CoreML-compatible phase computation
# ---------------------------------------------------------------------------

class SineGen:
    @staticmethod
    def _f02sine(self, f0_values):
        """CoreML-compatible phase: fractional freq -> pool -> cumsum -> interpolate -> sin."""
        val = f0_values / self.sampling_rate
        rad_values = val - torch.floor(val)

        if not self.flag_for_pulse:
            K = int(self.upsample_scale)
            x = rad_values.transpose(1, 2)
            x = F.avg_pool1d(x, kernel_size=K, stride=K)
            x = torch.cumsum(x, dim=2)
            x = x * (2.0 * torch.pi * K)
            x = F.interpolate(x, scale_factor=float(K),
                              mode='linear', align_corners=True)
            phase = x.transpose(1, 2)

            two_pi = 2.0 * torch.pi
            phase = phase - two_pi * torch.floor(phase / two_pi)
            sines = torch.sin(phase)

        return sines


# ---------------------------------------------------------------------------
# Patching functions
# ---------------------------------------------------------------------------

def _patch_snake_mul(model):
    """Replace pow(sin(x), 2) with sin(x) * sin(x) in Snake activations."""
    from kokoro.istftnet import AdaINResBlock1

    def _snake_mul_forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            s1 = torch.sin(a1 * xt)
            xt = xt + (1.0 / a1) * (s1 * s1)
            xt = c1(xt)
            xt = n2(xt, s)
            s2 = torch.sin(a2 * xt)
            xt = xt + (1.0 / a2) * (s2 * s2)
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _snake_mul_forward


def patch_sinegen_for_export(model):
    """Apply inlined SineGen to the model and return a phases helper.

    Makes the model deterministic by zeroing random phases and eliminating
    dead noise computations.
    """
    from kokoro.istftnet import SineGen as OriginalSineGen

    OriginalSineGen._f02sine = SineGen._f02sine

    def _zero_noise_forward(self, f0):
        fn = torch.multiply(f0, torch.FloatTensor(
            [[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        sine_waves = sine_waves * uv
        return sine_waves, uv, torch.zeros_like(sine_waves)
    OriginalSineGen.forward = _zero_noise_forward

    _patch_snake_mul(model)

    from kokoro.istftnet import SourceModuleHnNSF
    def _no_noise_source_forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, torch.zeros_like(uv), uv
    SourceModuleHnNSF.forward = _no_noise_source_forward

    def set_phases(module, phases):
        for m in module.modules():
            if isinstance(m, OriginalSineGen):
                m._external_phases = torch.zeros_like(phases)

    return set_phases


def patch_pack_padded_sequence():
    """Replace pack_padded_sequence/pad_packed_sequence with no-ops."""
    nn.utils.rnn.pack_padded_sequence = lambda x, lengths, **kw: x
    nn.utils.rnn.pad_packed_sequence = lambda x, **kw: (x, None)
