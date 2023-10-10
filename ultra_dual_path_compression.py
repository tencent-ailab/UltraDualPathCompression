# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: torchcpu
#     language: python
#     name: python3
# ---

# +
# Author: Hangting Chen
# Copyright: Tencent AI Lab
# Paper: Ultra Dual-Path Compression For
# Joint Echo Cancellation and Noise Suppression
# This code give the source of the ultra dual-path
# compression module and an example of usage.
# -

import torch
import torch.nn as nn
import numpy as np
import math


# # Define neural network modules


class WAVSTFT(nn.Module):
    def __init__(self, win_size=320):
        super(WAVSTFT, self).__init__()
        window = torch.from_numpy(np.hanning(win_size).astype(np.float32))
        self.window_size = window.shape[-1]
        self.hop_length = self.window_size // 2
        window = window.unsqueeze(0).unsqueeze(-1)
        divisor = torch.ones(1, 1, 1, self.window_size * 4)
        divisor = nn.functional.unfold(
            divisor, (1, self.window_size), stride=self.hop_length
        )
        divisor = divisor * window.pow(2.0)
        divisor = nn.functional.fold(
            divisor,
            (1, self.window_size * 4),
            (1, self.window_size),
            stride=self.hop_length,
        )[:, 0, 0, :]
        divisor = (
            divisor[0, self.window_size: 2 * self.window_size]
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("window", window)
        self.register_buffer("divisor", divisor)

    def magphase(self, complex_tensor: torch.Tensor):
        mag = complex_tensor.pow(2.0).sum(-1).pow(0.5 * 1.0)
        phase = torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])
        return mag, phase

    def add_window(self, x, divisor):
        out = x * self.window / divisor
        return out

    def frame(self, x):
        assert x.dim() == 2, x.shape
        out = x.unsqueeze(1).unsqueeze(1)
        out = nn.functional.pad(
            out, (self.window_size, self.window_size), "constant", 0
        )
        out = nn.functional.unfold(
            out, (1, self.window_size), stride=self.hop_length
        )  # B N T
        return out

    def overlap_and_add(self, x, length):
        assert x.dim() == 3, x.shape
        out = nn.functional.fold(
            x,
            (1, length + 2 * self.window_size),
            (1, self.window_size),
            stride=self.hop_length,
        )[:, 0, 0, :]
        out = out[:, self.window_size: -self.window_size]
        return out

    def rfft(self, x):
        assert x.dim() == 3, x.shape
        return torch.fft.rfft(x, dim=1)

    def irfft(self, x):
        assert x.dim() == 3, x.shape
        return torch.fft.irfft(x, dim=1)

    def STFT(self, x, return_aux=False):
        assert x.dim() == 2, x.shape
        out = self.frame(x)
        out = self.add_window(out, 1)
        out = self.rfft(out)
        if return_aux:
            mag, phase = self.magphase(torch.view_as_real(out))
            lps = torch.log(mag**2 + 1e-8)
            return out, mag, phase, lps
        else:
            return out

    def iSTFT(self, x, length):
        assert x.dim() == 3, x.shape
        out = self.irfft(x)
        out = self.add_window(out, self.divisor)
        out = self.overlap_and_add(out, length=length)
        return out


class TimeCompression(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, steps):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.dim4 = dim4
        self.steps = steps
        self.trans1 = nn.Conv2d(
            self.dim1 * self.steps, self.dim2, 1, bias=False
        )
        self.trans2 = nn.Conv2d(self.dim3, self.dim4, 1, bias=False)

    def forward(self, x, inverse):
        # x B C T F
        if inverse:
            B, C, T, F = x.shape
            x = (
                self.trans2(x)
                .reshape(B, -1, 1, T, F)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )  # B C T S F
            x = x.repeat(1, 1, 1, self.steps, 1).reshape(
                B, -1, T * self.steps, F
            )  # B C T*S F
            x = torch.nn.functional.pad(
                x, (0, 0, self.steps - 1, 0), "constant", 0
            )
            x = x[:, :, : -self.steps + 1, :]
            if self.pad > 0:
                x = x[:, :, self.pad:, :]
            return x
        else:
            B, C, T, F = x.shape
            if x.shape[-2] % self.steps == 0:
                self.pad = 0
            else:
                self.pad = self.steps - x.shape[-2] % self.steps
                x = torch.nn.functional.pad(
                    x, (0, 0, self.pad, 0), "constant", 0
                )
            x = (
                x.reshape(B, C, -1, self.steps, F)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )  # B C S T F
            x = x.reshape(B, C * self.steps, -1, F)  # B C*S T F
            return self.trans1(x)


class FreqCompression(nn.Module):
    def __init__(
        self, nfreq, nfilters, in_dim, hidden_dim, out_dim, sample_rate=16000
    ):
        super().__init__()
        self.nfreq = nfreq
        self.nfilters = nfilters
        self.sample_rate = sample_rate
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        mel_scale = "htk"

        all_freqs = torch.linspace(0, sample_rate // 2, nfreq)
        # calculate mel freq bins
        m_min = self._hz_to_mel(0, mel_scale=mel_scale)
        m_max = self._hz_to_mel(sample_rate / 2.0, mel_scale=mel_scale)

        m_pts = torch.linspace(m_min, m_max, self.nfilters + 2)
        f_pts = self._mel_to_hz(m_pts, mel_scale=mel_scale)
        self.bounds = [
            0,
        ]
        for freq_inx in range(1, len(f_pts) - 1):
            self.bounds.append(
                (all_freqs > f_pts[freq_inx]).float().argmax().item()
            )
        self.bounds.append(nfreq)
        self.trans1 = nn.ModuleList()
        self.trans2 = nn.ModuleList()
        for freq_inx in range(self.nfilters):
            self.trans1.append(
                nn.Linear(
                    (self.bounds[freq_inx + 2] - self.bounds[freq_inx])
                    * self.in_dim,
                    self.hidden_dim,
                    bias=False,
                )
            )
            self.trans2.append(
                nn.Conv1d(
                    self.hidden_dim,
                    (self.bounds[freq_inx + 2] - self.bounds[freq_inx])
                    * self.out_dim,
                    1,
                )
            )

    def _hz_to_mel(self, freq: float, mel_scale: str = "htk") -> float:
        r"""
        Source: https://pytorch.org/audio/stable/
        _modules/torchaudio/functional/functional.html
        Convert Hz to Mels.

        Args:
            freqs (float): Frequencies in Hz
            mel_scale (str, optional): Scale to use:
            ``htk`` or ``slaney``. (Default: ``htk``)

        Returns:
            mels (float): Frequency in Mels
        """

        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale should be one of "htk" or "slaney".')

        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (freq - f_min) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep

        return mels

    def _mel_to_hz(
        self, mels: torch.Tensor, mel_scale: str = "htk"
    ) -> torch.Tensor:
        """
        Source: https://pytorch.org/audio/stable/
        _modules/torchaudio/functional/functional.html
        Convert mel bin numbers to frequencies.

        Args:
            mels (torch.Tensor): Mel frequencies
            mel_scale (str, optional): Scale to use:
            ``htk`` or ``slaney``. (Default: ``htk``)

        Returns:
            freqs (torch.Tensor): Mels converted in Hz
        """

        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale should be one of "htk" or "slaney".')

        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0

        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(
            logstep * (mels[log_t] - min_log_mel)
        )

        return freqs

    def forward(self, x, inverse):
        if inverse:
            # B C T F
            out = torch.zeros(
                [x.shape[0], self.out_dim, self.nfreq, x.shape[2]],
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
            )
            for freq_inx in range(self.nfilters):
                out[
                    :, :, self.bounds[freq_inx]: self.bounds[freq_inx + 2], :
                ] = out[
                    :, :, self.bounds[freq_inx]: self.bounds[freq_inx + 2], :
                ] + self.trans2[
                    freq_inx
                ](
                    x[:, :, :, freq_inx]
                ).reshape(
                    x.shape[0], self.out_dim, -1, x.shape[-2]
                )
            out[:, :, self.bounds[1]: self.bounds[-2], :] = (
                out[:, :, self.bounds[1]: self.bounds[-2], :] / 2.0
            )
            out = out.permute(0, 1, 3, 2).contiguous().tanh()
            return out
        else:
            x = x.reshape(x.shape[0], self.in_dim, *x.shape[-2:])  # B C T F
            x = x.permute(0, 2, 1, 3).contiguous()  # B T C F
            x = torch.stack(
                [
                    self.trans1[freq_inx](
                        x[
                            :,
                            :,
                            :,
                            self.bounds[freq_inx]: self.bounds[freq_inx + 2],
                        ].flatten(start_dim=2)
                    )
                    for freq_inx in range(self.nfilters)
                ],
                -1,
            )  # B T C F
            x = x.permute(0, 2, 1, 3).contiguous()
            return x


class UltraDualPath(nn.Module):
    def __init__(
        self,
        nfreq,
        in_dim,
        hidden_dim,
        out_dim,
        freq_cprs_ratio,
        time_cprs_ratio,
    ):
        super(UltraDualPath, self).__init__()
        self.nfreq = nfreq
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.freq_cprs_ratio = freq_cprs_ratio
        self.time_cprs_ratio = time_cprs_ratio
        self.compress_modules = [
            TimeCompression(
                self.in_dim,
                self.in_dim * 2,
                self.out_dim,
                self.out_dim,
                time_cprs_ratio,
            ),
            FreqCompression(
                self.nfreq,
                self.nfreq // self.freq_cprs_ratio,
                self.in_dim * 2,
                self.hidden_dim,
                self.out_dim,
            ),
        ]

    def forward(self, x, inverse):
        out = x
        # print(out.shape)
        if inverse:
            for m in self.compress_modules[::-1]:
                out = m(out, inverse)
                # print(out.shape)
        else:
            for m in self.compress_modules:
                out = m(out, inverse)
                # print(out.shape)
        return out


# # An example of run

# ## Define model and input

# ## Define network modules
win_size = 320
wavSTFT = WAVSTFT(win_size)  # 16khz, 20ms/10ms -> 320 samples
ultraCompress = UltraDualPath(
    win_size // 2 + 1, 3 * 2, 48, 48, 4, 4
)  # 3ch*(real+imag)
# ## Define input
wav = torch.rand(2, 3, 16000)  # nbatch=2, nchannels=3, nsamples=16000

# ## Run

# organize input
B, C, N = wav.shape
spec = wavSTFT.STFT(wav.reshape(B * C, N))
spec = torch.view_as_real(spec.reshape(B, C, *spec.shape[-2:]))  # B C F T 2
spec = spec.permute(0, 1, 4, 3, 2)  # B C 2 T F
spec = spec.reshape(B, C * 2, *spec.shape[-2:])
print(spec.shape)  # nbatch, nchannels*2, nframes, nfreqs

# compress
latent = ultraCompress(spec, 0)
print(latent.shape)  # nbatch, hidden_dim, nframes, nfreqs

# decompress
output = ultraCompress(latent, 1)
print(output.shape)  # nbatch, out_dim, nframes, nfreqs
