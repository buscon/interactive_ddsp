#!/usr/bin/env python3
"""
Real-time streaming SensorDDSP server controlled over OSC from SuperCollider.

Receives OSC messages at 100 Hz (default UDP port 9000):
    /ddsp/sensor  <sensor: float 0..1> [sound_id: int] [note_hz: float]
    /ddsp/sound   <sound_id: int>
    /ddsp/note    <note_hz: float>       (0.0 = use f0_base for current sound)
    /ddsp/gain    <gain: float>
    /ddsp/reset                          (reset GRU state, touch_time, phases)

Audio is synthesized in 480-sample (10 ms @ 48 kHz) blocks and played via
sounddevice (PortAudio / PipeWire / JACK).

Usage:
    python3 realtime_ddsp.py --run <run_dir> --ckpt state_final.pth
    python3 realtime_ddsp.py --run runs/bakeoff_ddsp_poc/stageB_v8c_b --benchmark
"""
import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import (
    scale_function,
    remove_above_nyquist,
    amp_to_impulse_response,
)


# ---------------------------------------------------------------------------
# Minimal pure-Python OSC 1.0 UDP receiver (zero external dependencies)
# ---------------------------------------------------------------------------

def _parse_osc_message(data: bytes):
    """Parse a single OSC 1.0 message into (address, [args])."""
    def read_str(buf, offset):
        end = buf.index(b"\x00", offset)
        s = buf[offset:end].decode("utf-8", errors="replace")
        next_off = (end + 4) & ~3
        return s, next_off

    if not data or data[0:1] != b"/":
        return None, []
    address, off = read_str(data, 0)
    if off >= len(data) or data[off:off + 1] != b",":
        return address, []
    typetags, off = read_str(data, off)
    args = []
    for tag in typetags[1:]:
        if tag == "f":
            args.append(struct.unpack(">f", data[off:off + 4])[0])
            off += 4
        elif tag == "i":
            args.append(struct.unpack(">i", data[off:off + 4])[0])
            off += 4
        elif tag == "d":
            args.append(struct.unpack(">d", data[off:off + 8])[0])
            off += 8
        elif tag == "s":
            s, off = read_str(data, off)
            args.append(s)
        elif tag == "T":
            args.append(True)
        elif tag == "F":
            args.append(False)
    return address, args


def _iter_osc_messages(packet: bytes):
    """Yield (address, args) from an OSC message or #bundle."""
    if packet.startswith(b"#bundle\x00"):
        off = 16  # skip '#bundle\0' + 8-byte timetag
        while off + 4 <= len(packet):
            size = struct.unpack(">i", packet[off:off + 4])[0]
            off += 4
            if size <= 0 or off + size > len(packet):
                break
            yield from _iter_osc_messages(packet[off:off + size])
            off += size
    else:
        addr, args = _parse_osc_message(packet)
        if addr is not None:
            yield addr, args


# ---------------------------------------------------------------------------
# Uniformly partitioned FFT overlap-add reverb (real-time, causal)
# ---------------------------------------------------------------------------

class PartitionedReverb:
    """Real-time causal convolution of a frozen impulse response at block_size."""

    def __init__(self, ir_1d: np.ndarray, block_size: int):
        self.block_size = block_size
        self.fft_size = 1 << (2 * block_size - 1).bit_length()  # next pow2 >= 2*B
        n_parts = int(math.ceil(len(ir_1d) / block_size))
        self.n_parts = max(1, n_parts)

        ir_padded = np.zeros(self.n_parts * block_size, dtype=np.float32)
        ir_padded[:len(ir_1d)] = ir_1d.astype(np.float32)

        # Precompute FFT of each IR partition
        self.H = np.zeros((self.n_parts, self.fft_size // 2 + 1), dtype=np.complex64)
        for p in range(self.n_parts):
            seg = ir_padded[p * block_size:(p + 1) * block_size]
            self.H[p] = np.fft.rfft(seg, n=self.fft_size)

        self.reset()

    def reset(self):
        self.X = np.zeros((self.n_parts, self.fft_size // 2 + 1), dtype=np.complex64)
        self.ptr = 0
        self.overlap = np.zeros(self.fft_size - self.block_size, dtype=np.float32)

    def process(self, x_block: np.ndarray) -> np.ndarray:
        """Process one block of length block_size -> float32[block_size]."""
        B = self.block_size
        self.X[self.ptr] = np.fft.rfft(x_block, n=self.fft_size)

        # Circular multiply-accumulate across partitions
        # Ordered from newest (ptr) backwards
        idx = (self.ptr - np.arange(self.n_parts)) % self.n_parts
        Y = np.sum(self.X[idx] * self.H, axis=0)
        y_full = np.fft.irfft(Y, n=self.fft_size).astype(np.float32)

        out = y_full[:B] + self.overlap[:B]
        new_overlap = np.zeros_like(self.overlap)
        tail_len = len(self.overlap) - B
        if tail_len > 0:
            new_overlap[:tail_len] = self.overlap[B:]
        new_overlap += y_full[B:]
        self.overlap = new_overlap

        self.ptr = (self.ptr + 1) % self.n_parts
        return out


# ---------------------------------------------------------------------------
# Checkpoint loader (compatible with both V9 4-sound and V6-V8c 5-sound runs)
# ---------------------------------------------------------------------------

def load_sensor_ddsp(run_dir: str, ckpt_name: str, device: torch.device):
    """Load SensorDDSP model + metadata from a run directory."""
    config_path = os.path.join(run_dir, "config.yaml")
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(config_path) as f:
        run_config = yaml.safe_load(f)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model_cfg = dict(run_config["model"])

    # Determine n_sounds from checkpoint buffers if possible
    n_sounds = int(state_dict["f0_base"].shape[0])
    model_cfg["n_sounds"] = n_sounds

    # Build f0_base_hz placeholder (overwritten by state_dict)
    f0_override = model_cfg.pop("f0_base_hz_override", None)
    lfo_rate_cfg = model_cfg.pop("lfo_rate_base", None)
    lfo_mul_cfg = model_cfg.pop("lfo_depth_mul_base", None)
    lfo_add_cfg = model_cfg.pop("lfo_depth_add_base", None)

    if f0_override is not None and len(f0_override) == n_sounds:
        f0_base_hz = [float(v) for v in f0_override]
    else:
        f0_base_hz = [float(v) for v in state_dict["f0_base"].tolist()]
    model_cfg["f0_base_hz"] = f0_base_hz

    # Match attack_lag_s length to n_sounds if present in config or state_dict
    if "attack_lag_s" in state_dict:
        model_cfg["attack_lag_s"] = state_dict["attack_lag_s"].tolist()
    elif "attack_lag_s" in model_cfg and len(model_cfg["attack_lag_s"]) != n_sounds:
        model_cfg.pop("attack_lag_s")

    # Auto-detect touch_amp_head from checkpoint
    if "touch_amp_proj.weight" in state_dict:
        model_cfg["touch_amp_head"] = True
    else:
        model_cfg["touch_amp_head"] = False

    # Construct model (handle n_sounds != 4 default buffer size for attack_lag_s)
    if "attack_lag_s" not in model_cfg:
        model_cfg["attack_lag_s"] = [0.38] * n_sounds

    # Temporarily patch default lfo_* buffer lengths if n_sounds != 5
    model = SensorDDSP(**model_cfg)

    # Resize lfo_* buffers if checkpoint has different n_sounds (e.g. 4 vs 5)
    for buf_name in ("lfo_rate_base", "lfo_depth_mul_base", "lfo_depth_add_base"):
        if buf_name in state_dict and getattr(model, buf_name).shape != state_dict[buf_name].shape:
            model.register_buffer(buf_name, torch.zeros_like(state_dict[buf_name]))

    # Adapt f0_head input dimension for older V6/V7/V8c checkpoints (516 vs 517)
    ckpt_f0_in = state_dict["f0_head.weight"].shape[1]
    if model.f0_head.in_features != ckpt_f0_in:
        model.f0_head = nn.Linear(ckpt_f0_in, 1)

    model.load_state_dict(state_dict, strict=True)

    # Apply config LFO overrides if explicitly provided (e.g., V9 swarm fix)
    with torch.no_grad():
        if lfo_rate_cfg is not None and len(lfo_rate_cfg) == n_sounds:
            model.lfo_rate_base.copy_(torch.tensor(lfo_rate_cfg, dtype=torch.float32))
        if lfo_mul_cfg is not None and len(lfo_mul_cfg) == n_sounds:
            model.lfo_depth_mul_base.copy_(torch.tensor(lfo_mul_cfg, dtype=torch.float32))
        if lfo_add_cfg is not None and len(lfo_add_cfg) == n_sounds:
            model.lfo_depth_add_base.copy_(torch.tensor(lfo_add_cfg, dtype=torch.float32))

    model.to(device).eval()

    # Sound names (best effort from data root index.json, else defaults)
    sound_names = None
    data_root = (run_config.get("data") or {}).get("root")
    if data_root and os.path.exists(os.path.join(data_root, "index.json")):
        with open(os.path.join(data_root, "index.json")) as f:
            idx = json.load(f)
        sound_names = sorted(idx["sound_ids"], key=lambda k: idx["sound_ids"][k])
    if not sound_names or len(sound_names) != n_sounds:
        default_5 = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
        default_4 = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "05_moss_and_mist"]
        sound_names = default_4 if n_sounds == 4 else default_5[:n_sounds]

    return model, sound_names, run_config


# ---------------------------------------------------------------------------
# Stateful streaming wrapper around SensorDDSP
# ---------------------------------------------------------------------------

class StreamingSensorDDSP:
    """Block-by-block causal streaming inference for SensorDDSP."""

    def __init__(
        self,
        model: SensorDDSP,
        device: torch.device,
        touch_threshold: float = 0.98,
        use_reverb: bool = True,
        freeze_f0: bool = False,
    ):
        self.model = model
        self.device = device
        self.sr = model.sampling_rate
        self.block_size = model.block_size
        self.dt = self.block_size / float(self.sr)
        self.touch_threshold = float(touch_threshold)
        self.freeze_f0 = bool(freeze_f0)
        self.f0_uses_note = (model.f0_head.in_features == model.hidden_size + 5)

        # Precompute static tensors
        self.harmonics = torch.arange(
            1, model.n_harmonic + 1, dtype=torch.float32, device=device
        ).view(1, -1)  # (1, n_harmonic)
        self.ramp = torch.linspace(
            1.0 / self.block_size, 1.0, self.block_size, dtype=torch.float32, device=device
        ).view(-1, 1)  # (B, 1)

        # Precompute frozen reverb IR for real-time partitioned convolution
        self.use_reverb = bool(use_reverb)
        if self.use_reverb:
            with torch.no_grad():
                ir = model.reverb.build_impulse().squeeze(0).squeeze(-1).detach().cpu().numpy()
            self.reverb = PartitionedReverb(ir, self.block_size)
        else:
            self.reverb = None

        self.reset()

    def reset(self):
        """Reset all streaming state (GRU hidden, phases, touch timer, noise/reverb)."""
        self.h = None
        self.touch_time = 0.0
        self.osc_phase = 0.0
        self.lfo_phase = 0.0
        self.prev_f0 = None
        self.prev_amps = None
        self.prev_white = torch.zeros(1, 1, self.block_size, dtype=torch.float32, device=self.device)
        if self.reverb is not None:
            self.reverb.reset()

    @torch.inference_mode()
    def step(self, sensor_val: float, sound_id: int, note_hz: float = 0.0, gain: float = 1.0) -> np.ndarray:
        """
        Synthesize one audio block (block_size samples, float32) from current sensor state.
        """
        m = self.model
        sensor_val = float(min(max(sensor_val, 0.0), 1.0))
        sound_id = int(min(max(sound_id, 0), m.n_sounds - 1))

        # 1. Live touch_time state machine (mirrors dataset preprocessing)
        if sensor_val >= self.touch_threshold:
            touch_onset_val = 1.0 if self.touch_time == 0.0 else 0.0
            self.touch_time += self.dt
        else:
            self.touch_time = 0.0
            touch_onset_val = 0.0

        t_sec = self.touch_time
        t_feat0 = math.log1p(t_sec)
        t_feat1 = math.exp(-t_sec / 0.5) if t_sec > 0.0 else 0.0
        t_feat2 = min(max(t_sec, 0.0), 40.0)

        # Build (1, 1, *) tensors
        sensor_t = torch.tensor([[[sensor_val]]], dtype=torch.float32, device=self.device)
        touch_t = torch.tensor([[[t_feat0, t_feat1, t_feat2]]], dtype=torch.float32, device=self.device)
        onset_t = torch.tensor([[[touch_onset_val]]], dtype=torch.float32, device=self.device)
        sound_t = torch.tensor([sound_id], dtype=torch.long, device=self.device)
        note_t = torch.tensor([[[float(note_hz)]]], dtype=torch.float32, device=self.device)

        # 2. Encoder MLPs + stateful GRU + output MLP
        sensor_mlp = m.mlp_sensor(sensor_t)
        touch_mlp = m.mlp_touch(touch_t)
        emb = m.sound_emb(sound_t).unsqueeze(1)  # (1, 1, emb_dim)

        gru_in = torch.cat([sensor_mlp, touch_mlp, emb], dim=-1)
        gru_out, self.h = m.gru(gru_in, self.h)

        mlp_in = torch.cat([gru_out, sensor_t, touch_t, emb, onset_t], dim=-1)
        mlp_out = m.out_mlp(mlp_in)

        # 3. Heads + lagged fast_head
        harmonic_param = m.harmonic_head(mlp_out)
        noise_param = m.noise_head(mlp_out)

        t_sec_raw = torch.expm1(touch_t[..., :1].clamp(min=0))
        in_t = (t_sec_raw > 0).float()
        lag = m.attack_lag_s[sound_t].view(-1, 1, 1)
        t_d = t_sec_raw - lag
        gate = in_t * (t_d >= 0).float()
        t_dc = t_d.clamp(min=0)
        fast = torch.cat(
            [
                gate * torch.exp(-t_dc / 0.03),
                gate * torch.exp(-t_dc / 0.1),
                gate * torch.exp(-t_dc / 0.3),
            ],
            dim=-1,
        )
        fh = m.fast_head(fast)
        harmonic_param = harmonic_param + fh[..., :m.harmonic_head.out_features]
        noise_param = noise_param + fh[..., m.harmonic_head.out_features:]

        # 4. f0 + stateful hybrid LFO
        f0_base = m.f0_base[sound_t].view(1, 1, 1)
        note_f0 = torch.where(note_t > 0, note_t, f0_base)

        if self.freeze_f0:
            f0_deviation = torch.zeros_like(note_t)
        else:
            if self.f0_uses_note:
                f0_in = torch.cat([mlp_out, onset_t, touch_t, note_t], dim=-1)
            else:
                f0_in = torch.cat([mlp_out, onset_t, touch_t], dim=-1)
            f0_deviation = torch.clamp(m.f0_head(f0_in), -3.0, 3.0)

        f0 = note_f0 * (2.0 ** (m.max_f0_octaves * torch.tanh(f0_deviation)))

        in_touch = (touch_t[..., :1] > 0).float()
        rate_emb = m.lfo_rate_emb(sound_t).squeeze(-1)
        depth_mul = m.lfo_depth_mul_base[sound_t] + m.lfo_depth_mul_emb(sound_t).squeeze(-1)
        depth_add_emb = m.lfo_depth_add_emb(sound_t).squeeze(-1)

        t_sc = (t_sec_raw - lag).clamp(min=0) * in_t
        depth_add_t = (
            (0.5 * t_sc).clamp(max=20.0) + 0.1 * depth_add_emb.view(-1, 1, 1)
        ) * (m.lfo_depth_add_base[sound_t] > 0).float().view(-1, 1, 1)
        rate_t = 1.0 + (0.3 * t_sc).clamp(1.0, 30.0) + 0.1 * rate_emb.view(-1, 1, 1)

        self.lfo_phase = (self.lfo_phase + float(rate_t.item()) * self.dt) % 1.0
        sin_phase = math.sin(2.0 * math.pi * self.lfo_phase)

        f0 = f0 * (1.0 + in_touch * depth_mul.view(1, 1, 1) * sin_phase)
        f0 = f0 + in_touch * depth_add_t * sin_phase
        f0 = torch.clamp(f0, 20.0, 2000.0)

        # 5. Harmonic & noise scaling
        harmonic_param = scale_function(harmonic_param)
        total_amp = harmonic_param[..., :1]
        amplitudes = remove_above_nyquist(harmonic_param[..., 1:], f0, self.sr)
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
        amplitudes = amplitudes * total_amp

        noise_scaled = scale_function(noise_param - 5.0)

        if m.touch_amp_head_enabled:
            raw = m.touch_amp_proj(torch.cat([fast, emb], dim=-1))
            t_harm_amp = gate * F.softplus(raw[..., :1])
            t_harm_dist = F.softmax(raw[..., 1:1 + m.n_harmonic], dim=-1)
            t_harm = remove_above_nyquist(t_harm_dist * t_harm_amp, f0, self.sr)
            amplitudes = amplitudes + t_harm
            t_noise = gate * F.softplus(raw[..., 1 + m.n_harmonic:])
            noise_scaled = noise_scaled + t_noise

        # 6. Smooth linear interpolation across block_size + continuous-phase additive synth
        curr_f0 = f0.view(1, 1)                 # (1, 1)
        curr_amps = amplitudes.view(1, -1)      # (1, n_harmonic)
        if self.prev_f0 is None:
            self.prev_f0 = curr_f0
            self.prev_amps = curr_amps

        f0_up = self.prev_f0 + self.ramp * (curr_f0 - self.prev_f0)         # (B, 1)
        amps_up = self.prev_amps + self.ramp * (curr_amps - self.prev_amps) # (B, n_harmonic)
        self.prev_f0 = curr_f0
        self.prev_amps = curr_amps

        dphi = (2.0 * math.pi / self.sr) * f0_up
        omega = self.osc_phase + torch.cumsum(dphi, dim=0)
        self.osc_phase = float(omega[-1, 0].item()) % (2.0 * math.pi)
        harmonic = (torch.sin(omega * self.harmonics) * amps_up).sum(dim=-1)  # (B,)

        # 7. Causal filtered noise (continuous white noise convolved with current FIR)
        white = torch.rand(1, 1, self.block_size, dtype=torch.float32, device=self.device) * 2.0 - 1.0
        impulse = amp_to_impulse_response(noise_scaled, self.block_size)  # (1, 1, B)

        # Linear causal convolution using [prev_white, white] -> take second half
        x_cat = torch.cat([self.prev_white, white], dim=-1)  # (1, 1, 2B)
        k_pad = F.pad(impulse, (0, self.block_size))         # (1, 1, 2B)
        noise_out = torch.fft.irfft(
            torch.fft.rfft(x_cat) * torch.fft.rfft(k_pad), n=2 * self.block_size
        )[0, 0, self.block_size:]
        self.prev_white = white

        sig = (harmonic + noise_out).detach().cpu().numpy().astype(np.float32)

        # 8. Optional learned reverb
        if self.reverb is not None:
            sig = self.reverb.process(sig)

        # 9. Gain + soft limiter
        if gain != 1.0:
            sig = sig * float(gain)
        sig = np.tanh(sig).astype(np.float32)
        return sig


# ---------------------------------------------------------------------------
# Main: OSC listener + real-time audio callback (or benchmark mode)
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Real-time OSC-controlled SensorDDSP server")
    p.add_argument("--run", required=True, help="Run directory containing config.yaml and checkpoint")
    p.add_argument("--ckpt", default="state_final.pth", help="Checkpoint filename (default: state_final.pth)")
    p.add_argument("--port", type=int, default=9000, help="UDP port to listen for OSC (default: 9000)")
    p.add_argument("--host", default="127.0.0.1", help="Host/IP to bind OSC listener (default: 127.0.0.1)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device")
    p.add_argument("--sound", type=int, default=0, help="Initial sound_id (default: 0)")
    p.add_argument("--gain", type=float, default=1.0, help="Initial output gain (default: 1.0)")
    p.add_argument("--touch-threshold", type=float, default=0.98, help="Touch threshold (default: 0.98)")
    p.add_argument("--no-reverb", action="store_true", help="Disable learned reverb (dry output)")
    p.add_argument("--freeze-f0", action="store_true", help="Force f0_deviation = 0 (use note_hz / f0_base directly)")
    p.add_argument("--audio-device", default=None, help="sounddevice output device name or index")
    p.add_argument("--benchmark", action="store_true", help="Run 200-block offline latency benchmark and exit")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading SensorDDSP from {args.run}/{args.ckpt} on {device}...")
    model, sound_names, _ = load_sensor_ddsp(args.run, args.ckpt, device)
    engine = StreamingSensorDDSP(
        model=model,
        device=device,
        touch_threshold=args.touch_threshold,
        use_reverb=not args.no_reverb,
        freeze_f0=args.freeze_f0,
    )
    print(f"  Sample rate : {engine.sr} Hz, block_size: {engine.block_size} ({engine.dt * 1000:.1f} ms)")
    print(f"  Sounds ({len(sound_names)}): {', '.join(f'{i}={n}' for i, n in enumerate(sound_names))}")
    print(f"  f0_base (Hz): {[round(v, 1) for v in model.f0_base.tolist()]}")
    print(f"  f0_head     : {'FROZEN (note/base only)' if args.freeze_f0 else f'active (in={model.f0_head.in_features})'}")
    print(f"  Reverb      : {'enabled (partitioned FFT)' if engine.use_reverb else 'disabled (dry)'}")

    if args.benchmark:
        # Warmup
        for _ in range(10):
            engine.step(0.5, args.sound)
        engine.reset()
        times_ms = []
        peaks = []
        for i in range(200):
            # Sweep 0 -> 1 (with touch hold) -> 0
            s_val = 1.0 if 80 <= i < 150 else (i / 80.0 if i < 80 else max(0.0, 1.0 - (i - 150) / 50.0))
            t0 = time.perf_counter()
            blk = engine.step(s_val, args.sound, 0.0, args.gain)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            peaks.append(float(np.abs(blk).max()))
        arr = np.array(times_ms)
        print(
            f"\nBenchmark (200 blocks x {engine.dt * 1000:.1f} ms = {200 * engine.dt:.1f} s audio):"
            f"\n  Mean : {arr.mean():.2f} ms | Median: {np.median(arr):.2f} ms | "
            f"p99: {np.percentile(arr, 99):.2f} ms | Max: {arr.max():.2f} ms"
            f"\n  Budget: {engine.dt * 1000:.1f} ms -> {'OK (real-time)' if np.percentile(arr, 99) < engine.dt * 1000 else 'TOO SLOW'}"
            f"\n  Peak audio amplitude: {max(peaks):.4f} (NaN={any(math.isnan(p) for p in peaks)})"
        )
        return

    try:
        import sounddevice as sd
    except ImportError:
        print(
            "\nERROR: 'sounddevice' is not installed.\n"
            "Install it with:\n"
            "  pip install sounddevice\n"
            "(and on Ubuntu/Debian if PortAudio is missing: sudo apt install libportaudio2)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Shared control state updated by OSC thread, read by audio callback
    state = {
        "sensor": 0.0,
        "sound_id": args.sound,
        "note_hz": 0.0,
        "gain": args.gain,
        "reset_req": False,
        "msg_count": 0,
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    def osc_loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        sock.settimeout(0.2)
        print(f"Listening for OSC on udp://{args.host}:{args.port}")
        while not stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            for addr, osc_args in _iter_osc_messages(data):
                with lock:
                    state["msg_count"] += 1
                    if addr == "/ddsp/sensor" and len(osc_args) >= 1:
                        state["sensor"] = float(osc_args[0])
                        if len(osc_args) >= 2:
                            state["sound_id"] = int(osc_args[1])
                        if len(osc_args) >= 3:
                            state["note_hz"] = float(osc_args[2])
                    elif addr == "/ddsp/sound" and len(osc_args) >= 1:
                        state["sound_id"] = int(osc_args[0])
                    elif addr == "/ddsp/note" and len(osc_args) >= 1:
                        state["note_hz"] = float(osc_args[0])
                    elif addr == "/ddsp/gain" and len(osc_args) >= 1:
                        state["gain"] = float(osc_args[0])
                    elif addr == "/ddsp/reset":
                        state["reset_req"] = True
        sock.close()

    osc_thread = threading.Thread(target=osc_loop, daemon=True)
    osc_thread.start()

    # Audio callback
    underruns = [0]

    def audio_callback(outdata, frames, time_info, status):
        if status.output_underflow:
            underruns[0] += 1
        with lock:
            s_val = state["sensor"]
            sid = state["sound_id"]
            note = state["note_hz"]
            gain = state["gain"]
            do_reset = state["reset_req"]
            state["reset_req"] = False

        if do_reset:
            engine.reset()

        blk = engine.step(s_val, sid, note, gain)
        # Mono -> stereo output
        outdata[:, 0] = blk
        if outdata.shape[1] > 1:
            outdata[:, 1] = blk

    audio_dev = args.audio_device
    if audio_dev is not None and audio_dev.isdigit():
        audio_dev = int(audio_dev)

    print("Starting audio stream (Ctrl+C to stop)...")
    try:
        with sd.OutputStream(
            samplerate=engine.sr,
            blocksize=engine.block_size,
            channels=2,
            dtype="float32",
            device=audio_dev,
            callback=audio_callback,
        ):
            while True:
                time.sleep(0.5)
                with lock:
                    s_val = state["sensor"]
                    sid = state["sound_id"]
                    note = state["note_hz"]
                    nmsg = state["msg_count"]
                sname = sound_names[min(max(sid, 0), len(sound_names) - 1)]
                print(
                    f"\r[OSC:{nmsg:6d}] sound={sid}({sname})  sensor={s_val:0.3f}  "
                    f"touch={engine.touch_time:5.2f}s  note={note:5.1f}Hz  xruns={underruns[0]}   ",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        osc_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
