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
                 block_size=480, sound_emb_dim=16, max_f0_octaves=1.0):
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
        # Input: GRU_out (hidden) + raw sensor (1) + raw touch (1) + emb (sound_emb_dim)
        out_mlp_input = hidden_size + 1 + 1 + sound_emb_dim
        self.out_mlp = _mlp(out_mlp_input, hidden_size, hidden_size, n_layers=2)
        
        # Heads
        self.harmonic_head = nn.Linear(hidden_size, n_harmonic + 1)
        self.noise_head = nn.Linear(hidden_size, n_bands)
        self.f0_head = nn.Linear(hidden_size, 1)
        
        # Initialize f0_head bias to 0 so initial f0 = base
        self.f0_head.bias.data.zero_()
        
        # Reverb (using correct signature: Reverb(length, sampling_rate))
        self.reverb = Reverb(length=sampling_rate // 10, sampling_rate=sampling_rate)
    
    def forward(self, sensor, touch, sound_id):
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
        
        mlp_in = torch.cat([gru_out, sensor_raw, touch_raw, emb_skip], dim=-1)
        mlp_out = self.out_mlp(mlp_in)  # (B, T, hidden)
        
        # Heads
        harmonic_param = self.harmonic_head(mlp_out)  # (B, T, n_harmonic+1)
        noise_param = self.noise_head(mlp_out)        # (B, T, n_bands)
        f0_deviation = self.f0_head(mlp_out)          # (B, T, 1)
        
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

    def loss_fn(self, pred, target, scales, overlap):
        """
        Multiscale spectral loss.
        
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
