"""
SensorDDSP: DDSP model for proximity sensor conditioning.
Conditions on sensor value, touch_time, and sound_id.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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



def _onset_from_touch(touch):
    """touch: (B,T,>=1) with ch0 = log1p(t_sec). onset[t]=1 iff in-touch and within one frame of touch start.
    Fixes spurious onsets at chunk frame 0 (mid-touch chunk starts)."""
    t_ch = touch[..., :1]
    t_sec = torch.expm1(t_ch.clamp(min=0))  # raw seconds
    in_t = (t_sec > 0).float()
    return in_t * (t_sec <= 0.015).float()

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
                 tilt_init=True, onset_weighting=None, attack_lag_s=None,
                 w_lin=100.0, w_rms=1.0, onset_rms_factor=None,
                 touch_amp_head=False):
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
            onset_weighting: dict with keys:
                - enabled: bool
                - factor: float (k=5.0)
                - window_ms: int (150)
                - dilate_frames: int (1)
                - touch_rate_hz: int (100)
                - peak_ms / pre_ms / decay_tau_ms / dilate_ms: optional, passed through
                  (profile defaults apply when absent)
            attack_lag_s: optional list/tuple/tensor of per-sound attack lags in seconds
                (order canopy, forestPad, swarm, modularPercussions, moss); None keeps
                the built-in defaults.
            w_lin: linear spectral loss weight (default 100.0)
            w_rms: RMS envelope loss weight (default 1.0)
            onset_rms_factor: optional factor override for RMS envelope onset weighting
            touch_amp_head: if True, enables additive post-scale_function touch transient bypass
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
        self.w_mod_spec = 0.0   # v5 experimental losses default OFF (v5 run regressed: val 28.3 > silence 23.2)
        self.w_attack = 0.0
        self.tilt_init = tilt_init
        self.w_lin = float(w_lin)
        self.w_rms = float(w_rms)
        self.onset_rms_factor = float(onset_rms_factor) if onset_rms_factor is not None else None
        self.touch_amp_head_enabled = bool(touch_amp_head)
        
        # Onset weighting parameters
        if onset_weighting is None:
            self.onset_weighting = {
                'enabled': False,
                'factor': 5.0,
                'window_ms': 150,
                'dilate_frames': 1,
                'touch_rate_hz': 100
            }
        else:
            # Pass ALL keys through (peak_ms/pre_ms/decay_tau_ms were silently dropped before).
            # NOTE: this changes behaviour of old configs that set peak_ms/pre_ms (v8c ran with the defaults).
            self.onset_weighting = {
                'enabled': False,
                'factor': 5.0,
                'window_ms': 150,
                'dilate_frames': 1,
                'touch_rate_hz': 100
            }
            self.onset_weighting.update(onset_weighting)
        
        # Register f0_base as buffer (per sound)
        self.register_buffer(
            "f0_base",
            torch.tensor(f0_base_hz, dtype=torch.float32)
        )
        
        # Input MLPs for sensor (1->hidden) and touch (3->hidden, log1p/decay/clip features)
        self.mlp_sensor = _mlp(1, hidden_size, hidden_size, n_layers=2)
        self.mlp_touch = _mlp(3, hidden_size, hidden_size, n_layers=2)
        
        # Sound embedding
        self.sound_emb = nn.Embedding(n_sounds, sound_emb_dim)
        
        # per-sound LFO (vibrato/FM) gated by touch
        # Base rate/depth tables from SC sources (per sound: canopy, forestPad, swarm, percussion, moss)
        # canopy: rate=1+clip(0.3t,1,30), depth_add=0.5t clip 0..20 (additive FM)
        # swarm: rate=0.1+0.03t, depth_mul=(0.3+0.8t)% multiplicative
        # forestPad/moss/percussion: no LFO
        self.register_buffer("lfo_rate_base", torch.tensor([1.0, 0.0, 0.1, 0.0, 0.0], dtype=torch.float32))
        self.register_buffer("lfo_depth_mul_base", torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0], dtype=torch.float32))
        self.register_buffer("lfo_depth_add_base", torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32))
        
        # Residual embeddings for LFO (small init)
        self.lfo_rate_emb = nn.Embedding(n_sounds, 1)
        self.lfo_depth_mul_emb = nn.Embedding(n_sounds, 1)
        self.lfo_depth_add_emb = nn.Embedding(n_sounds, 1)
        nn.init.uniform_(self.lfo_rate_emb.weight, -0.01, 0.01)
        nn.init.uniform_(self.lfo_depth_mul_emb.weight, -0.01, 0.01)
        nn.init.uniform_(self.lfo_depth_add_emb.weight, -0.01, 0.01)
        
        # Input to GRU: sensor_mlp + touch_mlp + emb = hidden + hidden + sound_emb_dim
        gru_input_dim = hidden_size + hidden_size + sound_emb_dim
        self.gru = nn.GRU(gru_input_dim, hidden_size, batch_first=True)
        
        # Output MLP with skip connection
        # Input: GRU_out (hidden) + raw sensor (1) + raw touch (3, log1p/decay/clip) + emb (sound_emb_dim) + touch_onset (1)
        out_mlp_input = hidden_size + 1 + 3 + sound_emb_dim + 1
        self.out_mlp = _mlp(out_mlp_input, hidden_size, hidden_size, n_layers=2)

        # Heads
        self.harmonic_head = nn.Linear(hidden_size, n_harmonic + 1)
        self.noise_head = nn.Linear(hidden_size, n_bands)
        self.f0_head = nn.Linear(hidden_size + 1 + 3 + 1, 1)  # +1 touch_onset, +3 touch, +1 note
        # V8b: fast-decay touch features -> direct additive contribution to amp heads.
        # Zero-init so a V6 checkpoint (without fast_head keys) behaves identically at start.
        self.fast_head = nn.Linear(3, (n_harmonic + 1) + n_bands)
        with torch.no_grad():
            self.fast_head.weight.zero_()
            self.fast_head.bias.zero_()
        # V8c: per-sound attack lag (audio event lands ~+0.4s after the 0.98 crossing; Task 2 refines)
        # Default for 4 sounds (canopy, forestPad, swarm, moss); overridden by config
        self.register_buffer('attack_lag_s', torch.tensor([0.023, 0.026, 0.065, 0.025]))  # measured v6 lag (docs/V9_LAG_MEASUREMENT.md); overridden by config attack_lag_s
        if attack_lag_s is not None:
            self.set_attack_lag(attack_lag_s)

        # Optional additive post-scale_function touch transient bypass
        if self.touch_amp_head_enabled:
            self.touch_amp_proj = nn.Linear(3 + sound_emb_dim, 1 + n_harmonic + n_bands)
            with torch.no_grad():
                self.touch_amp_proj.weight.zero_()
                self.touch_amp_proj.bias.fill_(-6.0)
        else:
            self.touch_amp_proj = None

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
    
    def set_attack_lag(self, values):
        """Copy per-sound attack lags (s, sound_id order) in place into the attack_lag_s buffer.
        Accepts list/tuple/tensor of length n_sounds; keeps the buffer's dtype and device."""
        buf = self.attack_lag_s
        v = torch.as_tensor(values, dtype=buf.dtype, device=buf.device).reshape(-1)
        if v.numel() != self.n_sounds:
            raise ValueError(f"attack_lag_s needs {self.n_sounds} values, got {v.numel()}")
        with torch.no_grad():
            buf.copy_(v.detach())

    def _tilt_init(self):
        """Apply tilt initialization to heads."""
        # Multiply harmonic_head and noise_head weights by 0.1
        self.harmonic_head.weight.data.mul_(0.1)
        self.noise_head.weight.data.mul_(0.1)
        
        # f0_head: clamp pre-activation to [-3, 3] via weight scaling
        self.f0_head.weight.data.mul_(0.1)
        
        # harmonic_head bias: [-0.98, then -0.05*h for h in 0..n_harmonic-1]
        nh = self.n_harmonic
        b = torch.zeros(nh + 1)
        b[0] = -0.98
        b[1:] = -0.05 * torch.arange(nh).float()
        self.harmonic_head.bias.data.copy_(b)
        
        # noise_head bias: 2.0 - 0.15*k for k in 0..n_bands-1
        nb = self.n_bands
        self.noise_head.bias.data.copy_(2.0 - 0.15 * torch.arange(nb).float())
    

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # V8b/V9: allow checkpoints that predate fast_head or touch_amp_proj (e.g. V6 state_final.pth)
        for name, p in self.named_parameters():
            if (name.startswith('fast_head.') or name.startswith('touch_amp_proj.')) and (prefix + name) not in state_dict:
                state_dict[prefix + name] = p.detach().clone()
        for name, b in self.named_buffers():
            if name == 'attack_lag_s' and (prefix + name) not in state_dict:
                state_dict[prefix + name] = b.detach().clone()  # V8c: new buffer, default per-sound lag
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, sensor, touch, sound_id, touch_onset=None, note=None):
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
            # onset from the touch-seconds feature via shared helper (no false onsets at frame 0)
            touch_onset = _onset_from_touch(touch)

        # touch may arrive as (B,T,1) legacy normalised seconds/30 — expand to 3 features
        if touch.shape[-1] == 1:
            t_sec = touch * 30.0
            touch = torch.cat([torch.log1p(t_sec), (t_sec > 0) * torch.exp(-t_sec / 0.5), torch.clamp(t_sec, 0, 40)], dim=-1)

        # Process sensor and touch through MLPs
        sensor_mlp = self.mlp_sensor(sensor)  # (B, T, hidden)
        touch_mlp = self.mlp_touch(touch)      # (B, T, 3 -> hidden)
        
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
        touch_raw = touch    # (B, T, 3)
        emb_skip = emb       # (B, T, sound_emb_dim)
        
        mlp_in = torch.cat([gru_out, sensor_raw, touch_raw, emb_skip, touch_onset], dim=-1)
        mlp_out = self.out_mlp(mlp_in)  # (B, T, hidden)
        
        # Heads
        harmonic_param = self.harmonic_head(mlp_out)  # (B, T, n_harmonic+1)
        noise_param = self.noise_head(mlp_out)        # (B, T, n_bands)
        # V8c: fast-decay touch envelope -> direct additive amp contribution, DELAYED by attack lag
        t_sec_raw = torch.expm1(touch[..., :1].clamp(min=0))          # (B,T,1) seconds in touch
        in_t = (t_sec_raw > 0).float()
        lag = self.attack_lag_s[sound_id].view(-1, 1, 1)               # (B,1,1)
        t_d = t_sec_raw - lag                                           # time since attack (negative before lag / out of touch)
        gate = in_t * (t_d >= 0).float()                                # active only in-touch AND after lag
        t_dc = t_d.clamp(min=0)
        fast = torch.cat([gate * torch.exp(-t_dc / 0.03),
                          gate * torch.exp(-t_dc / 0.1),
                          gate * torch.exp(-t_dc / 0.3)], dim=-1)  # (B,T,3)
        fh = self.fast_head(fast)
        harmonic_param = harmonic_param + fh[..., :self.harmonic_head.out_features]
        noise_param = noise_param + fh[..., self.harmonic_head.out_features:]
        f0_deviation = self.f0_head(torch.cat([mlp_out, touch_onset, touch, note], dim=-1))  # (B, T, 1)
        f0_deviation = torch.clamp(f0_deviation, -3.0, 3.0)  # keep tanh out of saturation
        
        # Compute f0 per frame
        # f0 = f0_base[sound_id] * 2^(max_f0_octaves * tanh(f0_deviation))
        f0_base_sound = self.f0_base[sound_id]  # (B,)
        f0_base = f0_base_sound.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        # v4: absolute pitch from note input (0 = no touch -> base); residual deviation on top
        if note is None:
            note = torch.zeros_like(f0_base)
        note_f0 = torch.where(note > 0, note, f0_base)  # (B, T, 1)
        f0 = note_f0 * (2 ** (self.max_f0_octaves * torch.tanh(f0_deviation)))  # (B, T, 1)
        # v5: hybrid per-sound LFO with base tables + residual embeddings
        t_ch = touch[..., :1]  # log1p feature for touch detection
        in_touch = (t_ch > 0).float()  # (B, T, 1)
        
        # Per-sound base rate/depth + residual embeddings
        # lfo_rate_base is (5,) - per-sound scalar
        # lfo_rate_emb is (5, 1) - per-sound 1-dim embedding
        # sound_id is (B,) -> indexing gives (B,)
        rate_base = self.lfo_rate_base[sound_id]  # (B,)
        rate_emb = self.lfo_rate_emb(sound_id).squeeze(-1)  # (B, 1) -> (B,)
        rate = rate_base + rate_emb  # (B,), rate in Hz
        
        depth_mul_base = self.lfo_depth_mul_base[sound_id]  # (B,)
        depth_mul_emb = self.lfo_depth_mul_emb(sound_id).squeeze(-1)  # (B,)
        depth_mul = depth_mul_base + depth_mul_emb  # (B,), multiplicative depth
        
        depth_add_base = self.lfo_depth_add_base[sound_id]  # (B,)
        depth_add_emb = self.lfo_depth_add_emb(sound_id).squeeze(-1)  # (B,)
        depth_add = depth_add_base + depth_add_emb  # (B,), additive depth (Hz)
        
        # V8c: SC-formula vibrato with lagged timer: t_sc = clamp(t_sec - lag, 0) * in_touch
        # canopy SC: depth_add = clip(0.5*t, 0, 20); rate = 1 + clip(0.3*t, 1, 30)
        t_sc = (t_sec_raw - lag).clamp(min=0) * in_t                       # (B,T,1)
        depth_add_t = ((0.5 * t_sc).clamp(max=20.0) + 0.1 * depth_add_emb.view(-1, 1, 1)) * (self.lfo_depth_add_base[sound_id] > 0).float().view(-1, 1, 1)
        rate_t = 1.0 + (0.3 * t_sc).clamp(1.0, 30.0) + 0.1 * rate_emb.view(-1, 1, 1)  # (B,T,1) Hz
        phase = torch.cumsum(rate_t.squeeze(-1) / 100.0, dim=1)            # (B,T) cycles, time-varying rate
        sin_phase = torch.sin(2 * torch.pi * phase).unsqueeze(-1)           # (B,T,1)

        # Hybrid LFO: multiplicative depth + additive FM, gated by in_touch (no wobble before touch)
        f0 = f0 * (1.0 + in_touch * depth_mul.unsqueeze(-1).unsqueeze(-1) * sin_phase)
        f0 = f0 + in_touch * depth_add_t * sin_phase
        
        # Clamp f0 to valid range to prevent NaN from log2 or overflow
        f0 = torch.clamp(f0, 20.0, 2000.0)
        self._last_f0 = f0.detach() if not torch.is_grad_enabled() else f0  # stash for f0 loss
        
        # Apply scale function to harmonic params
        harmonic_param = scale_function(harmonic_param)
        total_amp = harmonic_param[..., :1]           # (B, T, 1)
        amplitudes = harmonic_param[..., 1:]          # (B, T, n_harmonic)
        
        # Normalise amplitudes
        amplitudes = remove_above_nyquist(amplitudes, f0, self.sampling_rate)
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
        amplitudes = amplitudes * total_amp

        # Filtered noise
        # noise_param has shape (B, T, n_bands) - one per frame
        # Apply scale function like original EngineDDSP: scale_function(noise_param - 5)
        # This ensures small initial amplitudes (shifts mean to ~0 before scaling)
        noise_scaled = scale_function(noise_param - 5)

        # Optional additive post-scale_function touch transient bypass
        if self.touch_amp_head_enabled:
            raw = self.touch_amp_proj(torch.cat([fast, emb], dim=-1))
            t_harm_amp = gate * F.softplus(raw[..., :1])
            t_harm_dist = F.softmax(raw[..., 1:1 + self.n_harmonic], dim=-1)
            t_harm = remove_above_nyquist(t_harm_dist * t_harm_amp, f0, self.sampling_rate)
            amplitudes = amplitudes + t_harm
            t_noise = gate * F.softplus(raw[..., 1 + self.n_harmonic:])
            noise_scaled = noise_scaled + t_noise
        
        # Upsample to audio rate
        amplitudes_up = upsample(amplitudes, self.block_size)  # (B, T*block_size, n_harmonic)
        f0_up = upsample(f0, self.block_size)                  # (B, T*block_size, 1)
        
        # Harmonic synthesis
        harmonic = harmonic_synth(f0_up, amplitudes_up, self.sampling_rate)  # (B, T*block_size, 1)
        
        # Create impulse responses for each frame
        impulse = amp_to_impulse_response(noise_scaled, self.block_size)  # (B, T, block_size)
        
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

    def estimate_f0_track(self, target, win=2048, fmin=50.0, fmax=500.0):
        """
        Self-supervised per-block f0 estimate from target audio via autocorrelation.
        target: (B, L) -> f0 (B, T) with T = L // block_size; NaN where unvoiced.
        """
        B, L = target.shape
        T = L // self.block_size
        x = target[:, :T * self.block_size].view(B, T, self.block_size)
        # widen window by unfolding neighbour blocks: use win samples centered per block
        pad = (win - self.block_size) // 2
        xp = torch.nn.functional.pad(target, (pad, pad + (win - self.block_size) - pad))
        # gather windows: (B, T, win)
        idx = torch.arange(T, device=target.device) * self.block_size + pad
        w = torch.stack([xp[:, i:i + win] for i in idx], dim=1)  # (B, T, win)
        w = w - w.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(w, n=2 * win, dim=-1)
        ac = torch.fft.irfft(spec * spec.conj(), n=2 * win, dim=-1)[..., :win]
        ac = ac / (ac[..., :1] + 1e-8)
        lo = max(int(self.sampling_rate / fmax), 1)
        hi = min(int(self.sampling_rate / fmin), win - 1)
        seg = ac[..., lo:hi]
        best = torch.argmax(seg, dim=-1) + lo  # (B, T)
        peak = torch.gather(ac, -1, best.unsqueeze(-1)).squeeze(-1)
        # Compute local std of ac peak over ±3 blocks for variance measure
        # Pad peak on the TIME dim: (B, T + 6)
        peak_padded = torch.nn.functional.pad(peak, (3, 3), mode='replicate')  # (B, T+6)
        # Unfold to get windows of ±3 blocks: (B, T, 7)
        local_windows = peak_padded.unfold(dimension=1, size=7, step=1)  # (B, T, 7)
        local_std = local_windows.std(dim=-1)  # (B, T)
        rms = x.pow(2).mean(dim=-1).sqrt()
        voiced = (peak > 0.6) & (rms > 1e-3) & (local_std < 0.1)
        f0 = self.sampling_rate / best.float()
        f0 = torch.where(voiced, f0, torch.full_like(f0, float('nan')))
        return f0

    def f0_regression_loss(self, f0_pred, target):
        """L1 in octaves between predicted and target-estimated f0 on voiced blocks."""
        f0_t = self.estimate_f0_track(target)  # (B, T), NaN unvoiced
        f0_p = f0_pred.squeeze(-1)  # (B, T)
        mask = (~torch.isnan(f0_t)) & (f0_t >= 20.0) & (f0_t <= 2000.0)
        if mask.sum() == 0:
            return (f0_p * 0.0).mean()
        f0_p_m = torch.clamp(torch.nan_to_num(f0_p[mask], nan=100.0, posinf=2000.0, neginf=20.0), 20.0, 2000.0)
        f0_t_m = torch.clamp(torch.nan_to_num(f0_t[mask], nan=100.0, posinf=2000.0, neginf=20.0), 20.0, 2000.0)
        err = (torch.log2(f0_p_m) - torch.log2(f0_t_m)).abs().mean()
        return err

    def loss_curriculum(self, pred, target, scales, overlap, step, ramp0=150, ramp1=400,
                        touch=None, sound_id=None, return_losses=False,
                        w_lin=None, w_rms=None, onset_rms_factor=None):
        """
        Curriculum loss with linear weight on spectral loss and RMS envelope loss.
        Optional onset weighting for multiscale spectral terms.

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
            return_losses: if True, return dict of individual loss components
            w_lin: optional override for linear spectral weight (defaults to self.w_lin)
            w_rms: optional override for RMS envelope weight (defaults to self.w_rms)
            onset_rms_factor: optional factor override for RMS envelope onset weighting

        Returns:
            scalar loss or (scalar loss, dict of losses)
        """
        eff_w_lin = self.w_lin if w_lin is None else float(w_lin)
        eff_w_rms = self.w_rms if w_rms is None else float(w_rms)
        eff_rms_factor = self.onset_rms_factor if onset_rms_factor is None else onset_rms_factor

        pred_stft = multiscale_fft(pred, scales, overlap)
        target_stft = multiscale_fft(target, scales, overlap)

        # Per-STFT-frame weight from sustained-touch mask (touch_time > 2 s)
        def frame_weights(s, n_frames, B, device):
            if touch is None:
                return None
            hop = int(s * (1 - overlap))
            block = self.block_size
            idx = torch.clamp((torch.arange(n_frames, device=device) * hop) // block, max=touch.shape[1] - 1)
            w = 1.0 + 0.5 * (touch[:, idx, 0] > float(math.log1p(2.0))).float()  # (B, frames)
            return w.unsqueeze(1)  # V8c: (B, 1, frames) to broadcast with (B, freqs, frames)

        lin_loss = 0.0
        lg_loss = 0.0
        wsum = 0.0
        for s, p_s, t_s in zip(scales, pred_stft, target_stft):
            w = frame_weights(s, p_s.shape[2], p_s.shape[0], p_s.device)  # V8c: shape[2]=frames (was shape[1]=freqs)
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

        # Onset-weighted spectral + RMS loss (V8b: one 100 Hz profile, shared; lin/lg weighted
        # separately so the 100x lin / ramped-w lg balance from V6 is preserved)
        if self.onset_weighting['enabled'] and touch is not None:
            w100 = self._onset_weight_profile_100hz(touch[..., :1])  # (B, T_touch, 1)

            lin_loss_w = 0.0
            lg_loss_w = 0.0
            for scale_idx, (p_s, t_s) in enumerate(zip(pred_stft, target_stft)):
                scale = scales[scale_idx]
                hop = int(scale * (1 - overlap))
                w_s = self._profile_to_stft(w100, p_s.shape[2], hop)  # (B, frames, 1)
                w_e = w_s.unsqueeze(1)  # (B, 1, frames, 1)
                wsum = w_e.sum().clamp(min=1e-6) * p_s.shape[1]  # V8c: divide by freq bins too -> mean over freqs
                d_lin = (p_s - t_s).abs()
                d_lg = (safe_log(p_s + 1e-5) - safe_log(t_s + 1e-5)).abs()
                lin_loss_w = lin_loss_w + (d_lin.unsqueeze(-1) * w_e).sum() / wsum
                lg_loss_w = lg_loss_w + (d_lg.unsqueeze(-1) * w_e).sum() / wsum
            lin_loss = lin_loss_w / len(scales)
            lg_loss = lg_loss_w / len(scales)

            # RMS envelope term: profile aligned to block grid
            w100_rms = self._onset_weight_profile_100hz(touch[..., :1], factor_override=eff_rms_factor) if eff_rms_factor is not None else w100
            T_rms = pred_log_rms.shape[1]
            if w100_rms.shape[1] >= T_rms:
                w_r = w100_rms[:, :T_rms, :]
            else:
                pad = torch.ones_like(w100_rms[:, :1, :]).expand(-1, T_rms - w100_rms.shape[1], -1)
                w_r = torch.cat([w100_rms, pad], dim=1)
            d_rms = (pred_log_rms - target_log_rms).abs()
            w_rms_grid = w_r.reshape(d_rms.shape)  # match (B,T) or (B,T,1)
            rms_loss = (d_rms * w_rms_grid).sum() / w_rms_grid.sum().clamp(min=1e-6)

        multiscale_loss = (lin_loss + lg_loss)
        total = eff_w_lin * lin_loss + w * lg_loss + eff_w_rms * rms_loss

        # v4: modulation-depth match on sustained-touch frames (touch_time > 2 s)
        if touch is not None and touch.shape[-1] >= 1:
            sustained = (touch[..., 0] > float(np.log1p(2.0))).float()  # (B, T)
            if sustained.sum() > 10:
                pe = self.frame_log_rms(pred).squeeze(-1)
                te = self.frame_log_rms(target).squeeze(-1)
                ws = sustained.sum(1) + 1e-8
                pm = (pe * sustained).sum(1) / ws
                tm = (te * sustained).sum(1) / ws
                ps = (((pe - pm.unsqueeze(1)) ** 2) * sustained).sum(1) / ws
                ts = (((te - tm.unsqueeze(1)) ** 2) * sustained).sum(1) / ws
                # Guard against zero variance (sqrt of near-zero can cause NaN propagation)
                ps = torch.clamp(ps, min=1e-8)
                ts = torch.clamp(ts, min=1e-8)
                total = total + 1.0 * w * (ps.sqrt() - ts.sqrt()).abs().mean()

        # v5: modulation spectrum loss (match dominant modulation frequency in 0.1-10 Hz band)
        if touch is not None and touch.shape[-1] >= 3:
            mod_spec = self.mod_spec_loss(pred, target, touch)
            total = total + getattr(self, 'w_mod_spec', 0.0) * w * mod_spec

        # v5: attack loss (short-window STFT weighted after touch_onset)
        # Derive touch_onset from touch (first channel is log1p(t) which > 0 iff t > 0)
        if touch is not None and touch.shape[-1] >= 3:
            touch_onset_from_touch = _onset_from_touch(touch)
            atk_loss = self.attack_loss(pred, target, touch_onset_from_touch, step)
            total = total + getattr(self, 'w_attack', 0.0) * w * atk_loss

        # Harmonic-comb loss for f0 supervision, ramped in with the log term, weight 0.0x (per review: octave-ambiguous)
        f0_reg_val = torch.tensor(0.0, device=pred.device)
        if sound_id is not None:
            comb = self.harmonic_comb_loss(pred, target, sound_id, scales, overlap)
            total = total + 0.0 * w * comb
            # Self-supervised f0 regression (target-derived f0 track), weight 1.0x ramped
            f0_pred = getattr(self, '_last_f0', None)
            if f0_pred is not None and f0_pred.shape[1] == target.shape[1] // self.block_size:
                f0_reg_loss = self.f0_regression_loss(f0_pred, target)
                f0_reg_val = f0_reg_loss
                total = total + 1.0 * w * f0_reg_loss

        if return_losses:
            details = {
                'multiscale': multiscale_loss,
                'lin': lin_loss,
                'lg': lg_loss,
                'rms': rms_loss,
                'f0_reg': f0_reg_val,
                'total': total,
            }
            return total, details
        
        return total

    def mod_spec_loss(self, pred, target, touch):
        """
        Phase-invariant modulation spectrum loss.
        Match dominant modulation frequency (0.1-10 Hz) in log-RMS envelope.
        
        Args:
            pred: (B, L) predicted audio
            target: (B, L) target audio
            touch: (B, T, 3) touch features at 100 Hz frame rate
            
        Returns:
            scalar loss (modulation spectrum mismatch)
        """
        # Sustained frames: touch_time > 2 s (first channel: log1p(t) > log1p(2))
        sustained = (touch[..., 0] > np.log1p(2.0)).float()  # (B, T)
        
        if sustained.sum() <= 10:
            return (pred * 0.0).mean()
        
        # Compute log-RMS envelopes
        pred_log_rms = self.frame_log_rms(pred)  # (B, T)
        target_log_rms = self.frame_log_rms(target)  # (B, T)
        
        B, T = pred_log_rms.shape
        
        # Detrend by subtracting mean on sustained frames
        # For non-sustained frames, use the mean of sustained frames for that batch item
        pred_sustained_mean = (pred_log_rms * sustained).sum(dim=1, keepdim=True) / (sustained.sum(dim=1, keepdim=True) + 1e-8)
        target_sustained_mean = (target_log_rms * sustained).sum(dim=1, keepdim=True) / (sustained.sum(dim=1, keepdim=True) + 1e-8)
        
        pred_detrended = pred_log_rms - pred_sustained_mean
        target_detrended = target_log_rms - target_sustained_mean
        
        # Zero-pad or truncate to power of 2 for FFT
        n_fft = min(T, 512)
        pred_fft = torch.fft.rfft(pred_detrended[:, :n_fft], n=n_fft)
        target_fft = torch.fft.rfft(target_detrended[:, :n_fft], n=n_fft)
        
        # Frequency bins at 100 Hz frame rate: d=0.01s per frame
        freq_bins = torch.fft.fftfreq(n_fft, d=0.01, device=pred.device)[:n_fft//2+1]
        
        # Mask for 0.1-10 Hz band
        mask_band = (freq_bins >= 0.1) & (freq_bins <= 10.0)
        
        if mask_band.sum() == 0:
            return (pred * 0.0).mean()
        
        # Energy in modulation band
        pred_energy = (pred_fft.real**2 + pred_fft.imag**2)[:, mask_band].sum() / (n_fft * B)
        target_energy = (target_fft.real**2 + target_fft.imag**2)[:, mask_band].sum() / (n_fft * B)
        
        # L1 mismatch
        return (pred_energy - target_energy).abs().mean()

    def attack_loss(self, pred, target, touch_onset, step):
        """
        Attack loss: short-window STFT with weighted frames after touch_onset.
        
        Args:
            pred: (B, L) predicted audio
            target: (B, L) target audio
            touch_onset: (B, T, 1) at 100 Hz frame rate
            step: current training step (0-500 for ramp)
            
        Returns:
            scalar loss
        """
        # Short window STFT: win=9600 (200ms), hop=2400 (50% overlap)
        win = 9600
        hop = 2400
        scales = [win]
        overlap = 1.0 - hop / win  # 0.75
        
        pred_stft = multiscale_fft(pred, scales, overlap)
        target_stft = multiscale_fft(target, scales, overlap)
        
        # multiscale_fft returns a list of (B, freqs, frames) tensors (PyTorch stft order)
        # We only have one scale, so we take the first element
        pred_stft = pred_stft[0]  # (B, freqs, frames)
        target_stft = target_stft[0]  # (B, freqs, frames)
        
        # Frame weight ramp: 1 -> 3 over steps 0-500
        w = min(1.0, max(0.0, step / 500.0))
        frame_weight = 1.0 + 2.0 * w  # 1.0 -> 3.0
        
        # Weight frames within 200ms (5 frames at 100 Hz) after touch_onset
        B, _, T_stft_frames = pred_stft.shape  # (B, freqs, frames)
        
        # touch_onset is at 100 Hz with T frames, STFT has T_stft_frames frames
        # We'll downsample touch_onset to match STFT frames
        T_touch = touch_onset.shape[1]
        
        # Compute downsampling factor - average touch_onset over STFT frame windows
        # STFT frame i (at time i * hop / sr) covers touch frames [i*hop/480, (i+1)*hop/480)
        # hop = 2400, so each STFT frame covers 2400/480 = 5 touch frames at 100 Hz
        
        # For simplicity, we'll downsample by taking every Nth touch frame
        # where N = T_touch / T_stft_frames (approximately 100/21 ≈ 5)
        if T_touch > T_stft_frames:
            # Downsample touch_onset
            # Touch frame i corresponds to STFT frame i * T_stft_frames / T_touch
            indices = torch.arange(T_stft_frames, device=pred.device) * T_touch // T_stft_frames
            touch_onset_up = touch_onset[:, indices, :]  # (B, T_stft_frames, 1)
        else:
            touch_onset_up = touch_onset
        
        # Create weight mask - frames with touch_onset or within 5 frames after
        weights = torch.ones(B, T_stft_frames, device=pred.device)
        
        # For each frame with onset, weight subsequent frames
        for offset in range(6):  # 0 to 5 frames after onset
            if offset == 0:
                weights = weights + touch_onset_up.squeeze(-1) * (frame_weight - 1.0)
            else:
                shifted = torch.nn.functional.pad(touch_onset_up.squeeze(-1)[:, :-offset], (offset, 0), value=0)
                weights = weights + shifted * (frame_weight - 1.0) / 2.0
        
        # Apply weights to STFT frames - need to match the frames dimension
        # pred_stft is (B, freqs, frames), weights is (B, frames)
        weights_expanded = weights.unsqueeze(1)  # (B, 1, frames)
        weighted_loss = (pred_stft - target_stft).abs() * weights_expanded
        weighted_loss = weighted_loss.mean()
        
        return w * weighted_loss

    def silence_loss(self, target, scales, overlap):
        """
        Compute the loss when pred is silence (zeros).
        Used as baseline for comparing model performance.
        """
        B, L = target.shape
        zeros = torch.zeros(B, L, dtype=target.dtype, device=target.device)
        return self.loss_fn(zeros, target, scales, overlap)

    def _compute_onset_weights(self, touch, stft_frame_rates):
        """
        Compute per-frame onset weights for multiscale loss.
        
        Args:
            touch: (B, T_touch, 1) float32, touch_time at 100 Hz
            stft_frame_rates: list of frame rates for each STFT scale
        
        Returns:
            List of weight tensors, one per STFT scale
            Each: (B, T_stft, 1) float32, normalized weights
        """
        # Derive binary touch_onset from touch_time
        # Touch onset = touch going from 0 to >0
        touch_onset = torch.zeros_like(touch)
        touch_onset[:, 1:, :] = ((touch[:, 1:, :] > 0) & (touch[:, :-1, :] == 0)).float()
        touch_onset[:, 0:1, :] = (touch[:, 0:1, :] > 0).float()
        
        weights_per_scale = []
        
        for scale_idx, frame_rate in enumerate(stft_frame_rates):
            # Convert 100 Hz touch to STFT frame rate
            # If frame_rate < 100: downsample with max-pooling
            # If frame_rate > 100: we need to upsample (predictive interpolation)
            # For simplicity, we'll use nearest neighbor upsampling
            
            if frame_rate <= 100:
                # Downsample
                downsample_factor = int(100 / frame_rate)
                if downsample_factor < 1:
                    downsample_factor = 1
                
                touch_onset_pooled = F.max_pool1d(
                    touch_onset.transpose(1, 2),  # (B, 1, T_touch)
                    kernel_size=downsample_factor,
                    stride=downsample_factor,
                ).transpose(1, 2).unsqueeze(-1)  # (B, T_stft, 1)
            else:
                # Upsample: STFT frames > touch frames
                # We'll use linear interpolation to go from 100 Hz to higher frame rate
                T_touch = touch_onset.shape[1]
                T_stft = int(T_touch * frame_rate / 100)
                
                # Resample: (B, 1, T_touch) -> (B, 1, T_stft)
                touch_onset_upsampled = F.interpolate(
                    touch_onset.transpose(1, 2),  # (B, 1, T_touch)
                    size=T_stft,
                    mode='nearest'
                )
                touch_onset_pooled = touch_onset_upsampled.transpose(1, 2).unsqueeze(-1)  # (B, T_stft, 1)
            
            # Create window in milliseconds
            # Window: 150ms after onset + 20ms pre-onset margin = 170ms total
            window_ms = 150  # Post-onset window
            pre_onset_ms = 20  # Pre-onset margin
            window_frames = int((window_ms + pre_onset_ms) * frame_rate / 1000)
            
            # Dilate by ±10ms (not ±1 frame)
            dilate_frames = int(10 * frame_rate / 1000)
            
            # Build weight mask
            # Window: pre_onset_ms (20ms) + window_ms (150ms) after onset = 170ms total
            # plus dilate_frames on each side
            weight = torch.zeros_like(touch_onset_pooled)
            T_stft = touch_onset_pooled.shape[1]
            
            for offset in range(-dilate_frames, window_frames + dilate_frames + 1):
                if offset == 0:
                    weight += touch_onset_pooled
                elif offset < 0:
                    # Shift left by -offset (touch_onset earlier in time)
                    pad_size = -offset
                    shifted = torch.zeros_like(touch_onset_pooled)
                    if T_stft - pad_size > 0:
                        shifted[:, pad_size:, :] = touch_onset_pooled[:, :T_stft - pad_size, :]
                    weight += shifted
                else:
                    # Shift right by offset (touch_onset later in time)
                    pad_size = offset
                    shifted = torch.zeros_like(touch_onset_pooled)
                    if T_stft - pad_size > 0:
                        shifted[:, :T_stft - pad_size, :] = touch_onset_pooled[:, pad_size:, :]
                    weight += shifted
            
            # Apply boost factor (k=5.0)
            k = self.onset_weighting['factor']  # From config, default 5.0
            frame_weights = 1.0 + (k - 1.0) * weight  # 1.0 baseline + boost at onsets
            
            # Normalize weights (important!)
            # We'll normalize per batch during loss computation
            weights_per_scale.append(frame_weights)
        
        return weights_per_scale

    def _onset_weight_profile_100hz(self, touch, factor_override=None):
        """One onset weight profile at the touch rate (100 Hz), causal decaying.

        w(t) = 1 + (k-1) * [ exp(-t/tau) response to onset for t in [0, window_ms]
                             + 0.5 * onset shifted back by pre_ms (flat pre-roll) ]
        Built with conv1d so the window points FORWARD in time (fixes backwards-shift bug).
        Returns (B, T, 1).
        """
        ow = self.onset_weighting
        k = ow.get('factor', 5.0) if factor_override is None else float(factor_override)
        window_ms = ow.get('window_ms', 600)
        tau_ms = ow.get('decay_tau_ms', 200)  # kept for back-compat
        peak_ms = ow.get('peak_ms', 350)  # gamma peak: measured audio event at ~330-360ms after crossing
        pre_ms = ow.get('pre_ms', 100)
        rate = ow.get('touch_rate_hz', 100)
        onset = _onset_from_touch(touch)  # (B,T,1)
        T = onset.shape[1]
        klen = int(window_ms * rate / 1000) + 1
        x = torch.arange(klen, device=onset.device).float()  # kernel position in frames
        tau_f = max(1.0, peak_ms * rate / 1000)  # gamma peak position in frames
        # gamma-shaped causal kernel: rises to peak at peak_ms, then decays (matches measured
        # event-locked energy: HP log-RMS jumps +10dB at ~330ms after the 0.98 crossing)
        kern = (x / tau_f) * torch.exp(1.0 - x / tau_f)
        kern = kern / kern.max().clamp(min=1e-8)  # normalize by PEAK so w reaches k at peak_ms (not sum!)
        kern = kern.flip(0)  # conv1d is cross-correlation: flip so kernel start aligns with onset
        prof = F.conv1d(onset.transpose(1, 2), kern.view(1, 1, -1), padding=klen - 1)[..., :T].transpose(1, 2)
        # flat pre-roll box: 0.5 boost over [onset-pre_ms, onset)
        pre_frames = int(pre_ms * rate / 1000)
        pre = torch.zeros_like(onset)
        if pre_frames > 0:
            box = torch.ones(1, 1, pre_frames, device=onset.device)
            padded = F.pad(onset.transpose(1, 2), (0, pre_frames))
            c = F.conv1d(padded, box)[..., 1:T + 1]
            pre = (c.transpose(1, 2) > 0).float() * 0.5
        w = 1.0 + (k - 1.0) * (prof + pre)
        return w.clamp(max=k)

    def _profile_to_stft(self, w100, stft_frames, hop_samples):
        """Max-pool the 100 Hz profile onto the STFT frame grid (fixes dropped onsets at coarse scales)."""
        block = self.block_size
        device = w100.device
        starts = (torch.arange(stft_frames, device=device) * hop_samples) // block
        span = max(1, (hop_samples + block - 1) // block + 1)
        idx = starts.unsqueeze(1) + torch.arange(span, device=device).unsqueeze(0)  # (F, span)
        idx = idx.clamp(max=w100.shape[1] - 1)
        return w100[:, idx, :].amax(dim=2)  # (B, F, 1)

    def loss(self, batch, return_losses=False):
        """
        Compute loss with optional onset weighting.
        
        NOTE: Onset weighting has been moved to `loss_curriculum()` for proper curriculum training.
        This method remains for backward compatibility but does not apply onset weighting.
        The onset weighting parameters are still recognized but ignored.
        
        Args:
            batch: dict with 'audio', 'sensor', 'touch', 'sound_id' keys
            return_losses: if True, return dict of individual loss components
            
        Returns:
            scalar loss or (scalar loss, dict of losses)
        """
        # Forward pass
        # Sensor is already (B, T, 1), touch is (B, T, 3)
        sensor = batch['sensor']
        touch = batch['touch']
        sound_id = batch['sound_id']
        note = batch.get('note', None)
        
        pred = self(sensor, touch, sound_id, note=note).squeeze(-1)
        target = batch['audio']
        
        # Scales from model config (use defaults if not set)
        scales = [4096, 2048, 1024, 512, 256, 128]
        overlap = 0.75
        
        # Compute multiscale loss (V6-style, without onset weighting)
        # Note: onset weighting is now in loss_curriculum() for V7 training
        multiscale_losses = []
        for i, scale in enumerate(scales):
            # Compute STFT for this scale
            pred_stft = multiscale_fft(pred, [scale], overlap)
            target_stft = multiscale_fft(target, [scale], overlap)
            
            # pred_stft is list of (B, frames, freqs) tensors
            pred_s = pred_stft[0]  # (B, frames, freqs)
            target_s = target_stft[0]  # (B, frames, freqs)
            
            # Per-frame loss: L1 on magnitude + L1 on log magnitude
            per_frame_loss = (pred_s - target_s).abs() + (safe_log(pred_s + 1e-5) - safe_log(target_s + 1e-5)).abs()
            # per_frame_loss: (B, frames, freqs)
            loss_i = per_frame_loss.mean()
            multiscale_losses.append(loss_i)
        
        multiscale_loss = sum(multiscale_losses) / len(multiscale_losses)
        
        # Compute other loss components
        result = {'multiscale': multiscale_loss}
        
        # RMS envelope loss
        pred_log_rms = self.frame_log_rms(pred)
        target_log_rms = self.frame_log_rms(target)
        rms_loss = (pred_log_rms - target_log_rms).abs().mean()
        result['rms'] = rms_loss
        
        total = multiscale_loss + rms_loss
        
        if return_losses:
            return total, result
        return total


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
        
        # Test onset weighting
        model_onsetw = SensorDDSP(f0_base, onset_weighting={'enabled': True, 'factor': 5.0})
        batch = {
            'sensor': sensor,
            'touch': torch.randn(B, T, 3),
            'sound_id': sound_id,
        }
        loss, losses = model_onsetw.loss(batch, return_losses=True)
        print(f"Onset-weighted loss: {loss.item():.6f}")
        print(f"Loss components: {losses}")
