#!/usr/bin/env python3
"""
Real-time streaming OSC audio server for SensorDDSP.

Mirrors the forward math of engine_ddsp.sensor_ddsp.SensorDDSP in single-frame
(B=1, T=1) causal streaming mode at 48 kHz / 480-sample blocks (10 ms), with:
  - Live touch_time state machine (threshold 0.98)
  - Stateful GRU hidden state, hybrid LFO phase, continuous-phase additive synth
  - Causal filtered noise via overlap-save FFT convolution
  - Causal uniformly partitioned FFT overlap-add learned Reverb (on by default)
  - Built-in pure-Python OSC 1.0 UDP receiver (zero external OSC dependencies)
  - Real-time stereo playback via sounddevice and offline --benchmark mode
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Ensure local repo root is importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import (
    amp_to_impulse_response,
    remove_above_nyquist,
    scale_function,
)


# ---------------------------------------------------------------------------
# Pure-Python OSC 1.0 UDP Parser
# ---------------------------------------------------------------------------

def _read_osc_string(buf: bytes, offset: int) -> Tuple[str, int]:
    """Read a null-terminated, 4-byte-padded OSC string."""
    end = buf.index(b"\x00", offset)
    text = buf[offset:end].decode("utf-8", errors="replace")
    next_offset = (end + 4) & ~3
    return text, next_offset


def parse_osc_packet(data: bytes) -> List[Tuple[str, List[Any]]]:
    """Parse an OSC 1.0 message or #bundle into a list of (address, args)."""
    if not data:
        return []

    if data.startswith(b"#bundle\x00"):
        # Header: "#bundle\0" (8 bytes) + timetag (8 bytes)
        messages: List[Tuple[str, List[Any]]] = []
        offset = 16
        n = len(data)
        while offset + 4 <= n:
            (elem_size,) = struct.unpack_from(">i", data, offset)
            offset += 4
            if elem_size <= 0 or offset + elem_size > n:
                break
            elem = data[offset : offset + elem_size]
            offset += elem_size
            messages.extend(parse_osc_packet(elem))
        return messages

    # Single OSC message
    address, offset = _read_osc_string(data, 0)
    if not address.startswith("/"):
        return []

    args: List[Any] = []
    if offset >= len(data):
        return [(address, args)]

    typetags, offset = _read_osc_string(data, offset)
    if not typetags.startswith(","):
        return [(address, args)]

    for tag in typetags[1:]:
        if tag == "f":
            (val,) = struct.unpack_from(">f", data, offset)
            offset += 4
            args.append(val)
        elif tag == "d":
            (val,) = struct.unpack_from(">d", data, offset)
            offset += 8
            args.append(val)
        elif tag == "i":
            (val,) = struct.unpack_from(">i", data, offset)
            offset += 4
            args.append(val)
        elif tag == "h":
            (val,) = struct.unpack_from(">q", data, offset)
            offset += 8
            args.append(val)
        elif tag == "s":
            val, offset = _read_osc_string(data, offset)
            args.append(val)
        elif tag == "b":
            (blob_len,) = struct.unpack_from(">i", data, offset)
            offset += 4
            blob = data[offset : offset + blob_len]
            offset += (blob_len + 3) & ~3
            args.append(blob)
        elif tag == "T":
            args.append(True)
        elif tag == "F":
            args.append(False)
        elif tag == "N":
            args.append(None)
        elif tag == "I":
            args.append(None)
        else:
            # Unknown OSC type tag; stop parsing remaining arguments
            break

    return [(address, args)]


# ---------------------------------------------------------------------------
# Checkpoint Loader
# ---------------------------------------------------------------------------

def _resolve_checkpoint_and_config(
    checkpoint_path: str, config_path: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Resolve checkpoint file path and optional config.yaml path."""
    if os.path.isdir(checkpoint_path):
        ckpt_file = os.path.join(checkpoint_path, "state_final.pth")
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(f"No state_final.pth found in {checkpoint_path}")
        if config_path is None:
            candidate = os.path.join(checkpoint_path, "config.yaml")
            if os.path.isfile(candidate):
                config_path = candidate
    else:
        ckpt_file = checkpoint_path
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
        if config_path is None:
            candidate = os.path.join(os.path.dirname(os.path.abspath(ckpt_file)), "config.yaml")
            if os.path.isfile(candidate):
                config_path = candidate
    return ckpt_file, config_path


def load_sensor_ddsp(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    config_path: Optional[str] = None,
) -> Tuple[SensorDDSP, dict]:
    """
    Load a SensorDDSP checkpoint (V6/V7/V8c 516-dim or V9 517-dim f0_head,
    4-sound or 5-sound buffers, with or without touch_amp_proj).
    """
    ckpt_file, cfg_file = _resolve_checkpoint_and_config(checkpoint_path, config_path)
    cfg: dict = {}
    if cfg_file is not None and os.path.isfile(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    raw_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
        state_dict = raw_ckpt["state_dict"]
    elif isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt and isinstance(raw_ckpt["model_state_dict"], dict):
        state_dict = raw_ckpt["model_state_dict"]
    else:
        state_dict = raw_ckpt

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    # Infer exact tensor dimensions from state_dict (authoritative over config)
    n_sounds = int(state_dict["sound_emb.weight"].shape[0])
    sound_emb_dim = int(state_dict["sound_emb.weight"].shape[1])
    hidden_size = int(state_dict["harmonic_head.weight"].shape[1])
    n_harmonic = int(state_dict["harmonic_head.weight"].shape[0]) - 1
    n_bands = int(state_dict["noise_head.weight"].shape[0])
    touch_amp_head = any(k.startswith("touch_amp_proj.") for k in state_dict)

    sampling_rate = int(model_cfg.get("sampling_rate", 48000))
    block_size = int(model_cfg.get("block_size", 480))
    max_f0_octaves = float(model_cfg.get("max_f0_octaves", 2.0))

    if "f0_base" in state_dict:
        f0_base_hz = state_dict["f0_base"].detach().cpu().tolist()
    elif "f0_base_hz_override" in model_cfg:
        f0_base_hz = list(model_cfg["f0_base_hz_override"])
    else:
        f0_base_hz = [110.0] * n_sounds

    # Pass attack_lag_s=None so SensorDDSP.__init__ does not fail on length mismatch
    model = SensorDDSP(
        f0_base_hz=f0_base_hz,
        n_sounds=n_sounds,
        hidden_size=hidden_size,
        n_harmonic=n_harmonic,
        n_bands=n_bands,
        sampling_rate=sampling_rate,
        block_size=block_size,
        sound_emb_dim=sound_emb_dim,
        max_f0_octaves=max_f0_octaves,
        tilt_init=False,
        attack_lag_s=None,
        touch_amp_head=touch_amp_head,
    )

    # Resize per-sound buffers to match state_dict shapes before strict loading
    for buf_name in ("attack_lag_s", "lfo_rate_base", "lfo_depth_mul_base", "lfo_depth_add_base"):
        if buf_name in state_dict:
            target_buf = state_dict[buf_name]
            cur_buf = getattr(model, buf_name)
            if cur_buf.shape != target_buf.shape:
                model.register_buffer(buf_name, torch.zeros_like(target_buf))
        elif buf_name == "attack_lag_s":
            # Legacy checkpoint without attack_lag_s: align default buffer length to n_sounds
            cur_buf = getattr(model, buf_name)
            if cur_buf.numel() != n_sounds:
                padded = torch.full((n_sounds,), 0.38, dtype=cur_buf.dtype)
                n_copy = min(cur_buf.numel(), n_sounds)
                padded[:n_copy] = cur_buf[:n_copy]
                model.register_buffer(buf_name, padded)

    # Adapt f0_head input dim (legacy 516-dim vs V9 517-dim) before strict loading
    f0_in_dim = int(state_dict["f0_head.weight"].shape[1])
    if model.f0_head.in_features != f0_in_dim:
        model.f0_head = nn.Linear(f0_in_dim, 1)

    model.load_state_dict(state_dict, strict=True)

    # Apply explicit config lfo_*_base overrides afterward if provided
    for buf_name in ("lfo_rate_base", "lfo_depth_mul_base", "lfo_depth_add_base"):
        if buf_name in model_cfg and model_cfg[buf_name] is not None:
            vals = list(model_cfg[buf_name])
            if len(vals) == n_sounds:
                with torch.no_grad():
                    getattr(model, buf_name).copy_(
                        torch.tensor(vals, dtype=getattr(model, buf_name).dtype)
                    )

    model.to(device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Causal Uniformly Partitioned FFT Overlap-Add Reverb
# ---------------------------------------------------------------------------

class PartitionedReverb:
    """
    Zero-latency causal uniformly partitioned FFT overlap-add convolution.
    Applies the model's precomputed 1-second learned impulse response.
    """

    def __init__(self, impulse: np.ndarray, block_size: int):
        self.block_size = int(block_size)
        self.fft_size = 2 * self.block_size
        ir = np.asarray(impulse, dtype=np.float32).reshape(-1)
        self.n_parts = max(1, int(math.ceil(len(ir) / self.block_size)))
        padded_len = self.n_parts * self.block_size
        if len(ir) < padded_len:
            ir = np.pad(ir, (0, padded_len - len(ir)))
        parts = ir.reshape(self.n_parts, self.block_size)
        self.H = np.fft.rfft(parts, n=self.fft_size, axis=1).astype(np.complex64)
        self.n_bins = self.H.shape[1]
        self.reset()

    def reset(self) -> None:
        self.fdl = np.zeros((self.n_parts, self.n_bins), dtype=np.complex64)
        self.overlap = np.zeros(self.block_size, dtype=np.float32)
        self.head = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Convolve one mono block of length block_size."""
        x = np.asarray(block, dtype=np.float32)
        X_cur = np.fft.rfft(x, n=self.fft_size).astype(np.complex64)

        self.head = (self.head - 1) % self.n_parts
        self.fdl[self.head] = X_cur

        tail_len = self.n_parts - self.head
        Y = np.sum(self.fdl[self.head :] * self.H[:tail_len], axis=0)
        if self.head > 0:
            Y = Y + np.sum(self.fdl[: self.head] * self.H[tail_len:], axis=0)

        y_full = np.fft.irfft(Y, n=self.fft_size).astype(np.float32)
        out = y_full[: self.block_size] + self.overlap
        self.overlap = y_full[self.block_size :].copy()
        return out


# ---------------------------------------------------------------------------
# Streaming Single-Frame SensorDDSP Engine
# ---------------------------------------------------------------------------

class StreamingSensorDDSP:
    """
    Single-frame (B=1, T=1) causal streaming wrapper around SensorDDSP.
    """

    def __init__(
        self,
        model: SensorDDSP,
        device: torch.device,
        touch_threshold: float = 0.98,
        freeze_f0: bool = False,
        use_reverb: bool = True,
        gain: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.block_size = int(model.block_size)
        self.sampling_rate = int(model.sampling_rate)
        self.dt = self.block_size / float(self.sampling_rate)
        self.n_sounds = int(model.n_sounds)
        self.n_harmonic = int(model.n_harmonic)
        self.touch_threshold = float(touch_threshold)
        self.freeze_f0 = bool(freeze_f0)
        self.use_reverb = bool(use_reverb)
        self.gain = float(gain)

        # Detect legacy 516-dim vs V9 517-dim f0_head
        self.f0_head_takes_note = (
            model.f0_head.in_features == model.hidden_size + 1 + 3 + 1
        )

        # Precompute constant tensors on device
        self.harm_indices = torch.arange(
            1, self.n_harmonic + 1, dtype=torch.float32, device=self.device
        )
        self.interp_alpha = torch.linspace(
            1.0 / self.block_size, 1.0, self.block_size, dtype=torch.float32, device=self.device
        )
        self.two_pi_over_sr = 2.0 * math.pi / float(self.sampling_rate)

        # Precompute learned reverb impulse response
        self.reverb: Optional[PartitionedReverb] = None
        if self.use_reverb:
            with torch.no_grad():
                ir = model.reverb.build_impulse().detach().reshape(-1).cpu().numpy()
            self.reverb = PartitionedReverb(ir, self.block_size)

        self.reset()

    def reset(self) -> None:
        """Reset all streaming states (GRU, touch timer, LFO, oscillators, noise, reverb)."""
        self.touch_time: float = 0.0
        self.h: Optional[torch.Tensor] = None
        self.lfo_phase: float = 0.0
        self.osc_phase: float = 0.0
        self.prev_f0: Optional[torch.Tensor] = None
        self.prev_amplitudes: Optional[torch.Tensor] = None
        self.prev_white = torch.zeros(self.block_size, dtype=torch.float32, device=self.device)
        if self.reverb is not None:
            self.reverb.reset()

    @torch.inference_mode()
    def process_block(
        self,
        sensor: float,
        sound_id: int = 0,
        note_hz: float = 0.0,
        gain: Optional[float] = None,
    ) -> np.ndarray:
        """
        Synthesize one causal audio block of shape (block_size,) float32.
        """
        m = self.model
        sid = max(0, min(int(sound_id), self.n_sounds - 1))
        sensor_val = float(sensor)

        # 1. Live touch_time state machine
        if sensor_val >= self.touch_threshold:
            touch_onset_val = 1.0 if self.touch_time == 0.0 else 0.0
            self.touch_time += self.dt
        else:
            self.touch_time = 0.0
            touch_onset_val = 0.0

        t = self.touch_time
        t_feat0 = math.log1p(t)
        t_feat1 = (1.0 if t > 0.0 else 0.0) * math.exp(-t / 0.5)
        t_feat2 = min(max(t, 0.0), 40.0)

        sensor_t = torch.tensor([[[sensor_val]]], dtype=torch.float32, device=self.device)
        touch_t = torch.tensor([[[t_feat0, t_feat1, t_feat2]]], dtype=torch.float32, device=self.device)
        touch_onset_t = torch.tensor([[[touch_onset_val]]], dtype=torch.float32, device=self.device)
        sound_id_t = torch.tensor([sid], dtype=torch.long, device=self.device)
        note_t = torch.tensor([[[float(note_hz)]]], dtype=torch.float32, device=self.device)

        # 2. Input MLPs + sound embedding + stateful GRU + output MLP
        sensor_mlp = m.mlp_sensor(sensor_t)
        touch_mlp = m.mlp_touch(touch_t)
        emb = m.sound_emb(sound_id_t).unsqueeze(1)  # (1, 1, sound_emb_dim)

        gru_in = torch.cat([sensor_mlp, touch_mlp, emb], dim=-1)
        gru_out, self.h = m.gru(gru_in, self.h)

        mlp_in = torch.cat([gru_out, sensor_t, touch_t, emb, touch_onset_t], dim=-1)
        mlp_out = m.out_mlp(mlp_in)

        # 3. Harmonic & noise heads + lagged fast_head
        harmonic_param = m.harmonic_head(mlp_out)
        noise_param = m.noise_head(mlp_out)

        t_sec_raw = torch.expm1(touch_t[..., :1].clamp(min=0.0))
        in_t = (t_sec_raw > 0.0).float()
        lag = m.attack_lag_s[sound_id_t].view(-1, 1, 1)
        t_d = t_sec_raw - lag
        gate = in_t * (t_d >= 0.0).float()
        t_dc = t_d.clamp(min=0.0)
        fast = torch.cat(
            [
                gate * torch.exp(-t_dc / 0.03),
                gate * torch.exp(-t_dc / 0.1),
                gate * torch.exp(-t_dc / 0.3),
            ],
            dim=-1,
        )
        fh = m.fast_head(fast)
        harmonic_param = harmonic_param + fh[..., : m.harmonic_head.out_features]
        noise_param = noise_param + fh[..., m.harmonic_head.out_features :]

        # 4. f0 head (516-dim legacy or 517-dim V9) + optional --freeze-f0
        f0_base = m.f0_base[sound_id_t].view(1, 1, 1)
        note_f0 = torch.where(note_t > 0.0, note_t, f0_base)

        if self.freeze_f0:
            f0 = note_f0
        else:
            if self.f0_head_takes_note:
                f0_in = torch.cat([mlp_out, touch_onset_t, touch_t, note_t], dim=-1)
            else:
                f0_in = torch.cat([mlp_out, touch_onset_t, touch_t], dim=-1)
            f0_deviation = torch.clamp(m.f0_head(f0_in), -3.0, 3.0)
            f0 = note_f0 * (2.0 ** (m.max_f0_octaves * torch.tanh(f0_deviation)))

        # 5. Stateful hybrid per-sound LFO
        in_touch = (touch_t[..., :1] > 0.0).float()
        rate_emb = m.lfo_rate_emb(sound_id_t).squeeze(-1)
        depth_mul = m.lfo_depth_mul_base[sound_id_t] + m.lfo_depth_mul_emb(sound_id_t).squeeze(-1)
        depth_add_emb = m.lfo_depth_add_emb(sound_id_t).squeeze(-1)

        t_sc = (t_sec_raw - lag).clamp(min=0.0) * in_t
        depth_add_t = (
            (0.5 * t_sc).clamp(max=20.0) + 0.1 * depth_add_emb.view(-1, 1, 1)
        ) * (m.lfo_depth_add_base[sound_id_t] > 0.0).float().view(-1, 1, 1)
        rate_t = 1.0 + (0.3 * t_sc).clamp(1.0, 30.0) + 0.1 * rate_emb.view(-1, 1, 1)

        self.lfo_phase = (self.lfo_phase + float(rate_t.item()) * self.dt) % 1.0
        sin_phase = math.sin(2.0 * math.pi * self.lfo_phase)

        f0 = f0 * (1.0 + in_touch * depth_mul.view(-1, 1, 1) * sin_phase)
        f0 = f0 + in_touch * depth_add_t * sin_phase
        f0 = torch.clamp(f0, 20.0, 2000.0)

        # 6. Scale harmonic and noise parameters + optional touch_amp_head
        harmonic_param = scale_function(harmonic_param)
        total_amp = harmonic_param[..., :1]
        amplitudes = harmonic_param[..., 1:]
        amplitudes = remove_above_nyquist(amplitudes, f0, self.sampling_rate)
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
        amplitudes = amplitudes * total_amp

        noise_scaled = scale_function(noise_param - 5.0)

        if m.touch_amp_head_enabled and m.touch_amp_proj is not None:
            raw = m.touch_amp_proj(torch.cat([fast, emb], dim=-1))
            t_harm_amp = gate * F.softplus(raw[..., :1])
            t_harm_dist = F.softmax(raw[..., 1 : 1 + self.n_harmonic], dim=-1)
            t_harm = remove_above_nyquist(t_harm_dist * t_harm_amp, f0, self.sampling_rate)
            amplitudes = amplitudes + t_harm
            t_noise = gate * F.softplus(raw[..., 1 + self.n_harmonic :])
            noise_scaled = noise_scaled + t_noise

        # 7. Smooth linear interpolation + continuous-phase additive oscillator bank
        cur_f0 = f0.reshape(())  # scalar tensor
        cur_amp = amplitudes.reshape(self.n_harmonic)

        if self.prev_f0 is None or self.prev_amplitudes is None:
            prev_f0 = cur_f0
            prev_amp = cur_amp
        else:
            prev_f0 = self.prev_f0
            prev_amp = self.prev_amplitudes

        f0_up = prev_f0 + self.interp_alpha * (cur_f0 - prev_f0)  # (B,)
        amp_up = prev_amp.unsqueeze(0) + self.interp_alpha.unsqueeze(1) * (cur_amp - prev_amp).unsqueeze(0)

        dphi = self.two_pi_over_sr * f0_up
        phase = self.osc_phase + torch.cumsum(dphi, dim=0)
        self.osc_phase = float(phase[-1].item()) % (2.0 * math.pi)

        omegas = phase.unsqueeze(1) * self.harm_indices.unsqueeze(0)
        harmonic = (torch.sin(omegas) * amp_up).sum(dim=-1)  # (B,)

        self.prev_f0 = cur_f0
        self.prev_amplitudes = cur_amp

        # 8. Causal filtered noise: concat [prev_white, white], FFT-convolve, take [block_size:]
        white = torch.rand(self.block_size, dtype=torch.float32, device=self.device) * 2.0 - 1.0
        white_cat = torch.cat([self.prev_white, white], dim=0)  # (2*B,)
        self.prev_white = white

        impulse = amp_to_impulse_response(noise_scaled, self.block_size).reshape(self.block_size)
        kernel_padded = F.pad(impulse, (0, self.block_size))  # (2*B,)
        noise_conv = torch.fft.irfft(
            torch.fft.rfft(white_cat) * torch.fft.rfft(kernel_padded),
            n=2 * self.block_size,
        )
        noise = noise_conv[self.block_size :]  # causal second half

        signal_mono = (harmonic + noise).detach().cpu().numpy().astype(np.float32)

        # 9. Causal learned Reverb (on by default)
        if self.reverb is not None:
            signal_mono = self.reverb.process(signal_mono)

        # 10. Output gain + tanh soft limiter
        eff_gain = self.gain if gain is None else float(gain)
        out = np.tanh(signal_mono * eff_gain).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Thread-safe OSC Control State & UDP Server
# ---------------------------------------------------------------------------

@dataclass
class ControlState:
    sensor: float = 0.0
    sound_id: int = 0
    note_hz: float = 0.0
    gain: float = 1.0
    reset_pending: bool = False


class OSCUDPReceiver:
    """Background UDP thread receiving OSC 1.0 messages and updating ControlState."""

    def __init__(
        self,
        host: str,
        port: int,
        state: ControlState,
        lock: threading.Lock,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.state = state
        self.lock = lock
        self.verbose = verbose
        self._stop_event = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.25)
        self._thread = threading.Thread(target=self._run, name="osc-udp-rx", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._sock.close()
        except OSError:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _dispatch(self, address: str, args: List[Any]) -> None:
        if self.verbose:
            print(f"[OSC] {address} {args}", flush=True)
        with self.lock:
            if address == "/ddsp/sensor" and len(args) >= 1:
                self.state.sensor = float(args[0])
                if len(args) >= 2:
                    self.state.sound_id = int(args[1])
                if len(args) >= 3:
                    self.state.note_hz = float(args[2])
            elif address == "/ddsp/sound" and len(args) >= 1:
                self.state.sound_id = int(args[0])
            elif address == "/ddsp/note" and len(args) >= 1:
                self.state.note_hz = float(args[0])
            elif address == "/ddsp/gain" and len(args) >= 1:
                self.state.gain = float(args[0])
            elif address == "/ddsp/reset":
                self.state.reset_pending = True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                for addr, args in parse_osc_packet(data):
                    self._dispatch(addr, args)
            except Exception as exc:
                if self.verbose:
                    print(f"[OSC] parse error: {exc}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Benchmark & Live Audio Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    engine: StreamingSensorDDSP,
    n_blocks: int = 200,
    sound_id: int = 0,
    note_hz: float = 0.0,
) -> dict:
    """
    Run a deterministic 0 -> 1 -> touch -> 0 sensor sweep and report block latency.
    """
    # Warmup
    for _ in range(20):
        engine.process_block(0.0, sound_id=sound_id, note_hz=note_hz)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    engine.reset()

    # Build 200-block sweep: 25% ramp up, 50% sustained touch (>=0.98), 25% ramp down
    n_up = max(1, n_blocks // 4)
    n_touch = max(1, n_blocks // 2)
    n_down = max(1, n_blocks - n_up - n_touch)
    sweep = np.concatenate(
        [
            np.linspace(0.0, 0.97, n_up, dtype=np.float32),
            np.ones(n_touch, dtype=np.float32),
            np.linspace(0.97, 0.0, n_down, dtype=np.float32),
        ]
    )

    times_ms = np.empty(len(sweep), dtype=np.float64)
    peak_amp = 0.0
    sq_sum = 0.0
    total_samples = 0
    has_nan_or_inf = False

    is_cuda = engine.device.type == "cuda"
    for i, s_val in enumerate(sweep):
        if is_cuda:
            torch.cuda.synchronize(engine.device)
        t0 = time.perf_counter()
        block = engine.process_block(float(s_val), sound_id=sound_id, note_hz=note_hz)
        if is_cuda:
            torch.cuda.synchronize(engine.device)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0

        if not np.all(np.isfinite(block)):
            has_nan_or_inf = True
        b_peak = float(np.max(np.abs(block)))
        if b_peak > peak_amp:
            peak_amp = b_peak
        sq_sum += float(np.dot(block, block))
        total_samples += len(block)

    budget_ms = engine.dt * 1000.0
    rms = math.sqrt(sq_sum / max(1, total_samples))
    stats = {
        "device": str(engine.device),
        "reverb": engine.use_reverb,
        "freeze_f0": engine.freeze_f0,
        "blocks": len(sweep),
        "budget_ms": budget_ms,
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "max_ms": float(np.max(times_ms)),
        "peak_amp": peak_amp,
        "rms": rms,
        "finite": not has_nan_or_inf,
    }

    print(
        f"Benchmark ({stats['blocks']} blocks, device={stats['device']}, "
        f"reverb={'ON' if stats['reverb'] else 'OFF'}, "
        f"freeze_f0={stats['freeze_f0']}):"
    )
    print(
        f"  Latency: mean={stats['mean_ms']:.3f} ms | median={stats['median_ms']:.3f} ms | "
        f"p99={stats['p99_ms']:.3f} ms | max={stats['max_ms']:.3f} ms "
        f"(budget={stats['budget_ms']:.2f} ms)"
    )
    print(
        f"  Audio:   peak={stats['peak_amp']:.4f} | rms={stats['rms']:.4f} | "
        f"NaN/Inf={'FAIL' if not stats['finite'] else 'OK (0)'}"
    )
    return stats


def run_server(
    engine: StreamingSensorDDSP,
    host: str,
    port: int,
    initial_sound: int = 0,
    initial_note: float = 0.0,
    initial_gain: float = 1.0,
    audio_device: Optional[str | int] = None,
    verbose: bool = False,
) -> None:
    """Start the OSC UDP receiver and sounddevice real-time OutputStream."""
    import sounddevice as sd

    # Warmup PyTorch/CUDA kernels before entering real-time audio callback
    for _ in range(10):
        engine.process_block(0.0, sound_id=initial_sound, note_hz=initial_note, gain=initial_gain)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    engine.reset()

    state = ControlState(
        sensor=0.0,
        sound_id=initial_sound,
        note_hz=initial_note,
        gain=initial_gain,
    )
    lock = threading.Lock()
    receiver = OSCUDPReceiver(host, port, state, lock, verbose=verbose)
    receiver.start()

    stop_event = threading.Event()

    def _handle_sigint(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    def audio_callback(outdata, frames, _time_info, status):
        if status and verbose:
            print(f"[AUDIO] {status}", file=sys.stderr, flush=True)
        with lock:
            s_val = state.sensor
            sid = state.sound_id
            note = state.note_hz
            g = state.gain
            do_reset = state.reset_pending
            state.reset_pending = False

        if do_reset:
            engine.reset()

        mono = engine.process_block(s_val, sound_id=sid, note_hz=note, gain=g)
        if frames == len(mono):
            outdata[:, 0] = mono
            outdata[:, 1] = mono
        else:
            n = min(frames, len(mono))
            outdata[:n, 0] = mono[:n]
            outdata[:n, 1] = mono[:n]
            if frames > n:
                outdata[n:, :] = 0.0

    print(
        f"Streaming SensorDDSP on {engine.device} | OSC udp://{host}:{port} | "
        f"sr={engine.sampling_rate} block={engine.block_size} | "
        f"reverb={'ON' if engine.use_reverb else 'OFF'} | freeze_f0={engine.freeze_f0}",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)

    try:
        with sd.OutputStream(
            samplerate=engine.sampling_rate,
            blocksize=engine.block_size,
            device=audio_device,
            channels=2,
            dtype="float32",
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                time.sleep(0.1)
    finally:
        receiver.stop()
        print("Server stopped.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time streaming OSC audio server for SensorDDSP."
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (with state_final.pth) or .pth file.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional explicit config.yaml path (auto-detected next to checkpoint if omitted).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto).",
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="OSC UDP bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=9000, help="OSC UDP bind port (default: 9000).")
    p.add_argument("--sound", type=int, default=0, help="Initial sound_id (default: 0).")
    p.add_argument("--note", type=float, default=0.0, help="Initial note_hz (0 = use f0_base).")
    p.add_argument("--gain", type=float, default=1.0, help="Initial output gain before tanh limiter.")
    p.add_argument(
        "--touch-threshold",
        type=float,
        default=0.98,
        help="Sensor threshold for touch_time onset (default: 0.98).",
    )
    p.add_argument(
        "--freeze-f0",
        action="store_true",
        help="Force f0_deviation=0 so f0 = note_f0 directly (useful for V9 checkpoints with drifted f0_head).",
    )
    p.add_argument(
        "--no-reverb",
        action="store_true",
        help="Disable learned partitioned FFT reverb (enabled by default).",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run offline 200-block sweep benchmark instead of live audio playback.",
    )
    p.add_argument(
        "--benchmark-blocks",
        type=int,
        default=200,
        help="Number of blocks for --benchmark mode (default: 200).",
    )
    p.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Optional sounddevice output device index or name.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List available sounddevice audio devices and exit.",
    )
    p.add_argument("--verbose", action="store_true", help="Log incoming OSC messages.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --list-devices is set.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        # Avoid OpenMP thread-pool wakeup spikes on small B=1, T=1 matmuls
        torch.set_num_threads(1)

    model, _ = load_sensor_ddsp(args.checkpoint, device=device, config_path=args.config)
    engine = StreamingSensorDDSP(
        model=model,
        device=device,
        touch_threshold=args.touch_threshold,
        freeze_f0=args.freeze_f0,
        use_reverb=not args.no_reverb,
        gain=args.gain,
    )

    if args.benchmark:
        stats = run_benchmark(
            engine,
            n_blocks=args.benchmark_blocks,
            sound_id=args.sound,
            note_hz=args.note,
        )
        return 0 if stats["finite"] else 1

    audio_dev: Optional[str | int] = args.audio_device
    if audio_dev is not None and audio_dev.isdigit():
        audio_dev = int(audio_dev)

    run_server(
        engine=engine,
        host=args.host,
        port=args.port,
        initial_sound=args.sound,
        initial_note=args.note,
        initial_gain=args.gain,
        audio_device=audio_dev,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
