"""
SensorDDSP: DDSP model for proximity sensor conditioning.
Conditions on sensor value, touch_time, and sound_id.
"""
import math
import torch
import torch.nn as nn
from .synth import (
    scale_function, remove_above_nyquist, upsample,
    harmonic_synth, amp_to_impulse_response, fft_convolve,
    Reverb, multiscale_fft, safe_log
)


def _mlp(in_dim, hidden, out_dim, n_layers=2):
    """Custom MLP: Linear + LayerNorm + LeakyReLU."""
    channels = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        layers.append(nn.LayerNorm(channels[i + 1]))
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


class SensorDDSP(nn.Module):
    """
    DDSP model conditioned on proximity sensor, touch_time, and sound_id.
    
    Architecture:
    - sensor and touch each go through their own MLP
    - sound_id goes through embedding, broadcast over time
    - Concat [mlp_sensor, mlp_touch, emb] -> GRU -> output MLP with skip
    - Heads: harmonic (n_harmonic+1), noise bands (n_bands), f0 deviation (1)
    - f0 = f0_base[sound_id] * 2^(max_f0_octaves * tanh(f0_head))
    """
    
    def __init__(self, f0_base_hz, n_sounds=5, hidden_size=512,
                 n_harmonic=100, n_bands=65, sampling_rate=48000,
                 block_size=480, sound_emb_dim=16, max_f0_octaves=1.0,
                 tilt_init=True):
        """
        Args:
            f0_base_hz: list of base f0 per sound ID (len = n_sounds)
            n_sounds: number of distinct sounds
            hidden_size: hidden dimension for MLPs and GRU
            n_harmonic: number of harmonic partials
            n_bands: number of noise filter bands
            sampling_rate: audio sample rate
            block_size: audio samples per frame
            sound_emb_dim: sound embedding dimension
            max_f0_octaves: max f0 deviation in octaves (tanh output * this)
            tilt_init: if True, apply tilt initialization to heads
        """
        super().__init__()
        
        self.n_sounds = n_sounds
        self.hidden_size = hidden_size
        self.n_harmonic = n_harmonic
        self.n_bands = n_bands
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.sound_emb_dim = sound_emb_dim
        self.max_f0_octaves = max_f0_octaves
        self.tilt_init = tilt_init
        
        # Register f0_base as buffer (per sound)
        self.register_buffer(
            "f0_base",
            torch.tensor(f0_base_hz, dtype=torch.float32)
        )
        
        # Input MLPs for sensor and touch (each: 1 -> hidden)
        self.mlp_sensor = _mlp(1, hidden_size, hidden_size, n_layers=2)
        self.mlp_touch = _mlp(1, hidden_size, hidden_size, n_layers=2)
        
        # Sound embedding
        self.sound_emb = nn.Embedding(n_sounds, sound_emb_dim)
        
        # Input to GRU: sensor_mlp + touch_mlp + emb = hidden + hidden + sound_emb_dim
        gru_input_dim = hidden_size + hidden_size + sound_emb_dim
        self.gru = nn.GRU(gru_input_dim, hidden_size, batch_first=True)
        
        # Output MLP with skip connection
        # Input: GRU_out (hidden) + raw sensor (1) + raw touch (1) + emb (sound_emb_dim) + touch_onset (1)
        out_mlp_input = hidden_size + 1 + 1 + sound_emb_dim + 1
        self.out_mlp = _mlp(out_mlp_input, hidden_size, hidden_size, n_layers=2)

        # Heads
        self.harmonic_head = nn.Linear(hidden_size, n_harmonic + 1)
        self.noise_head = nn.Linear(hidden_size, n_bands)
        self.f0_head = nn.Linear(hidden_size + 1, 1)  # +1 for touch_onset
        
        # Initialize f0_head bias to 0 so initial f0 = base
        self.f0_head.bias.data.zero_()
        
        # Apply tilt initialization if requested
        if tilt_init:
            self._tilt_init()
        
        # Reverb: 1 s impulse response as in the original EngineDDSP. initial_wet=-5
        # (sigmoid ~ 0.007) starts almost dry: the original default (wet=0 -> 0.5)
        # amplifies the untrained output ~25x (measured RMS 0.84 vs 0.034 dry,
        # target RMS ~0.06). The wet level is still learned.
        self.reverb = Reverb(length=sampling_rate, sampling_rate=sampling_rate,
                             initial_wet=-5)
    
    def _tilt_init(self):
        """Apply tilt initialization to heads."""
        # Multiply harmonic_head and noise_head weights by 0.1
        self.harmonic_head.weight.data.mul_(0.1)
        self.noise_head.weight.data.mul_(0.1)
        
        # harmonic_head bias: [-0.98, then -0.05*h for h in 0..n_harmonic-1]
        nh = self.n_harmonic
        b = torch.zeros(nh + 1)
        b[0] = -0.98
        b[1:] = -0.05 * torch.arange(nh).float()
        self.harmonic_head.bias.data.copy_(b)
        
        # noise_head bias: 2.0 - 0.15*k for k in 0..n_bands-1
        nb = self.n_bands
        self.noise_head.bias.data.copy_(2.0 - 0.15 * torch.arange(nb).float())
    
    def forward(self, sensor, touch, sound_id, touch_onset=None):
        """
        Forward pass.
        
        Args:
            sensor: (B, T, 1) float32, proximity sensor value 0..1
            touch: (B, T, 1) float32, normalised touch_time
            sound_id: (B,) int64, sound ID
        
        Returns:
            audio: (B, T*block_size, 1) float32
        """
        B, T, _ = sensor.shape

        # Derive binary touch_onset from touch_time if not provided:
        # onset[t] = 1 iff touch[t] > 0 and touch[t-1] == 0 (first frame: onset if touch[0] > 0)
        if touch_onset is None:
            touch_onset = torch.zeros_like(touch)
            touch_onset[..., 1:, :] = ((touch[..., 1:, :] > 0) & (touch[..., :-1, :] == 0)).float()
            touch_onset[..., 0, :] = (touch[..., 0, :] > 0).float()

        # Process sensor and touch through MLPs
        sensor_mlp = self.mlp_sensor(sensor)  # (B, T, hidden)
        touch_mlp = self.mlp_touch(touch)      # (B, T, hidden)
        
        # Sound embedding broadcast over time
        emb = self.sound_emb(sound_id)         # (B, sound_emb_dim)
        emb = emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, sound_emb_dim)
        
        # Concat for GRU input
        gru_in = torch.cat([sensor_mlp, touch_mlp, emb], dim=-1)  # (B, T, 2*hidden + sound_emb_dim)
        
        # GRU
        gru_out, _ = self.gru(gru_in)  # (B, T, hidden)
        
        # Output MLP with skip connection
        # Skip: raw sensor, raw touch, emb
        sensor_raw = sensor  # (B, T, 1)
        touch_raw = touch    # (B, T, 1)
        emb_skip = emb       # (B, T, sound_emb_dim)
        
        mlp_in = torch.cat([gru_out, sensor_raw, touch_raw, emb_skip, touch_onset], dim=-1)
        mlp_out = self.out_mlp(mlp_in)  # (B, T, hidden)
        
        # Heads
        harmonic_param = self.harmonic_head(mlp_out)  # (B, T, n_harmonic+1)
        noise_param = self.noise_head(mlp_out)        # (B, T, n_bands)
        f0_deviation = self.f0_head(torch.cat([mlp_out, touch_onset], dim=-1))  # (B, T, 1)
        
        # Compute f0 per frame
        # f0 = f0_base[sound_id] * 2^(max_f0_octaves * tanh(f0_deviation))
        f0_base_sound = self.f0_base[sound_id]  # (B,)
        f0_base = f0_base_sound.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        f0 = f0_base * (2 ** (self.max_f0_octaves * torch.tanh(f0_deviation)))  # (B, T, 1)
        
        # Apply scale function to harmonic params
        harmonic_param = scale_function(harmonic_param)
        total_amp = harmonic_param[..., :1]           # (B, T, 1)
        amplitudes = harmonic_param[..., 1:]          # (B, T, n_harmonic)
        
        # Normalise amplitudes
        amplitudes = remove_above_nyquist(amplitudes, f0, self.sampling_rate)
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
        amplitudes = amplitudes * total_amp
        
        # Upsample to audio rate
        amplitudes_up = upsample(amplitudes, self.block_size)  # (B, T*block_size, n_harmonic)
        f0_up = upsample(f0, self.block_size)                  # (B, T*block_size, 1)
        
        # Harmonic synthesis
        harmonic = harmonic_synth(f0_up, amplitudes_up, self.sampling_rate)  # (B, T*block_size, 1)
        
        # Filtered noise
        # noise_param has shape (B, T, n_bands) - one per frame
        # Apply scale function like original EngineDDSP: scale_function(noise_param - 5)
        # This ensures small initial amplitudes (shifts mean to ~0 before scaling)
        noise_param = scale_function(noise_param - 5)
        
        # Create impulse responses for each frame
        impulse = amp_to_impulse_response(noise_param, self.block_size)  # (B, T, block_size)
        
        # Generate noise (B, T, block_size) - one noise segment per frame
        noise = torch.rand(B, T, self.block_size, dtype=harmonic.dtype, device=harmonic.device) * 2 - 1
        
        # Convolve
        noise = fft_convolve(noise, impulse)  # (B, T, block_size)
        
        # Reshape to (B, T*block_size, 1)
        noise = noise.reshape(B, -1, 1)
        
        signal = harmonic + noise
        
        # Reverb
        signal = self.reverb(signal)
        
        return signal  # (B, T*block_size, 1)

    @staticmethod
    def frame_log_rms(x):
        """Compute log RMS over each 480-sample frame: log(RMS + 1e-4)."""
        # x: (B, L) or (B, L, 1)
        if x.dim() == 3:
            x = x.squeeze(-1)
        # Reshape to (B, n_frames, 480)
        B, L = x.shape
        n_frames = L // 480
        x_frames = x[:, :n_frames * 480].reshape(B, n_frames, 480)
        rms = x_frames.pow(2).mean(-1).sqrt()  # (B, n_frames)
        return torch.log(rms + 1e-4)

    def loss_fn(self, pred, target, scales, overlap):
        """
        Multiscale spectral loss (comparable reporting loss).
        
        Args:
            pred: (B, L) predicted audio
            target: (B, L) target audio
            scales: list of STFT window sizes
            overlap: overlap factor for STFT
        
        Returns:
            scalar loss
        """
        # multiscale_fft returns list of (B, frames, freqs) tensors
        pred_stft = multiscale_fft(pred, scales, overlap)
        target_stft = multiscale_fft(target, scales, overlap)
        
        loss = 0.0
        for p_s, t_s in zip(pred_stft, target_stft):
            # p_s, t_s: (B, frames, freqs)
            loss = loss + (p_s - t_s).abs().mean() + (safe_log(p_s) - safe_log(t_s)).abs().mean()
        
        return loss

    def harmonic_comb_loss(self, pred, target, sound_id, scales, overlap):
        """
        Harmonic-comb loss: penalize log-magnitude mismatch restricted to bins at
        harmonics of the per-sound base f0. Supervises the f0/harmonic structure
        without external pitch labels. Weight applied by caller.
        """
        pred_stft = multiscale_fft(pred, scales, overlap)
        target_stft = multiscale_fft(target, scales, overlap)
        f0 = self.f0_base[sound_id]  # (B,)
        loss = 0.0
        for s, p_s, t_s in zip(scales, pred_stft, target_stft):
            n_bins = p_s.shape[-1]
            bin_hz = self.sampling_rate / s
            B = p_s.shape[0]
            comb = torch.zeros(B, 1, n_bins, device=pred.device)
            for b in range(B):
                f0b = float(f0[b])
                h = 1
                while h * f0b < self.sampling_rate / 2:
                    bin_idx = int(round(h * f0b / bin_hz))
                    if bin_idx < n_bins:
                        comb[b, 0, bin_idx] = 1.0
                    h += 1
            p_l = safe_log(p_s + 1e-5)
            t_l = safe_log(t_s + 1e-5)
            diff = (p_l - t_l).abs() * comb
            denom = comb.sum() * p_s.shape[1] + 1e-8
            loss = loss + diff.sum() / denom
        return loss / len(scales)

    def loss_curriculum(self, pred, target, scales, overlap, step, ramp0=150, ramp1=400, touch=None, sound_id=None):
        """
        Curriculum loss with linear weight on spectral loss and RMS envelope loss.

        Args:
            pred: (B, L) predicted audio
            target: (B, L) target audio
            scales: list of STFT window sizes
            overlap: overlap factor for STFT
            step: current training step
            ramp0: start of ramp for log spectral loss weight
            ramp1: end of ramp for log spectral loss weight
            touch: optional (B, T_blocks, 1) touch_time at block rate (100 Hz); frames with
                touch_time > 2 s get +0.5x weight on the spectral terms (sustained-touch emphasis)
            sound_id: optional (B,) for the harmonic-comb loss

        Returns:
            scalar loss
        """
        pred_stft = multiscale_fft(pred, scales, overlap)
        target_stft = multiscale_fft(target, scales, overlap)

        # Per-STFT-frame weight from sustained-touch mask (touch_time > 2 s)
        def frame_weights(s, n_frames, B, device):
            if touch is None:
                return None
            hop = int(s * (1 - overlap))
            block = self.block_size
            idx = torch.clamp((torch.arange(n_frames, device=device) * hop) // block, max=touch.shape[1] - 1)
            w = 1.0 + 0.5 * (touch[:, idx, 0] > 2.0).float()  # (B, frames)
            return w.unsqueeze(-1)  # (B, frames, 1)

        lin_loss = 0.0
        lg_loss = 0.0
        wsum = 0.0
        for s, p_s, t_s in zip(scales, pred_stft, target_stft):
            w = frame_weights(s, p_s.shape[1], p_s.shape[0], p_s.device)
            if w is not None:
                d_lin = (p_s - t_s).abs()
                d_lg = (safe_log(p_s + 1e-5) - safe_log(t_s + 1e-5)).abs()
                lin_loss = lin_loss + (d_lin * w).sum() / (w.sum() * p_s.shape[-1] + 1e-8)
                lg_loss = lg_loss + (d_lg * w).sum() / (w.sum() * p_s.shape[-1] + 1e-8)
                wsum = wsum + 1.0
            else:
                lin_loss = lin_loss + (p_s - t_s).abs().mean()
                lg_loss = lg_loss + (safe_log(p_s + 1e-5) - safe_log(t_s + 1e-5)).abs().mean()

        # Weight for log spectral loss (ramps from 0 to 1)
        w = min(1.0, max(0.0, (step - ramp0) / (ramp1 - ramp0)))

        # RMS envelope loss
        pred_log_rms = self.frame_log_rms(pred)
        target_log_rms = self.frame_log_rms(target)
        rms_loss = (pred_log_rms - target_log_rms).abs().mean()

        total = 100.0 * lin_loss + w * lg_loss + 1.0 * rms_loss

        # Harmonic-comb loss for f0 supervision, ramped in with the log term, weight 0.2x
        if sound_id is not None:
            comb = self.harmonic_comb_loss(pred, target, sound_id, scales, overlap)
            total = total + 0.2 * w * comb

        return total

    def silence_loss(self, target, scales, overlap):
        """
        Compute the loss when pred is silence (zeros).
        Used as baseline for comparing model performance.
        """
        B, L = target.shape
        zeros = torch.zeros(B, L, dtype=target.dtype, device=target.device)
        return self.loss_fn(zeros, target, scales, overlap)


if __name__ == "__main__":
    # Quick test
    f0_base = [100.0, 150.0, 200.0, 250.0, 300.0]
    model = SensorDDSP(f0_base)
    
    B, T = 2, 100
    sensor = torch.randn(B, T, 1)
    touch = torch.randn(B, T, 1)
    sound_id = torch.randint(0, 5, (B,))
    
    with torch.no_grad():
        output = model(sensor, touch, sound_id)
        print(f"Input shapes: sensor={sensor.shape}, touch={touch.shape}, sound_id={sound_id.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: {output.min().item():.4f} to {output.max().item():.4f}")
        print(f"Has NaN: {torch.isnan(output).any().item()}")
        print(f"Has Inf: {torch.isinf(output).any().item()}")
        
        # Test tilt init
        model_tilt = SensorDDSP(f0_base, tilt_init=True)
        print(f"Tilt init harmonic_head weight range: {model_tilt.harmonic_head.weight.min().item():.6f} to {model_tilt.harmonic_head.weight.max().item():.6f}")
        print(f"Tilt init harmonic_head bias: {model_tilt.harmonic_head.bias[:5].tolist()}")
