#!/usr/bin/env python
"""
SensorDDSP training script.
Trains the sensor DDSP model on aligned sensor + audio data.
"""
import argparse
import os
import sys
import json
import yaml
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf

# Add the project to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_ddsp.sensor_data import AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import multiscale_fft, safe_log


class FixedChunkDataset(torch.utils.data.Dataset):
    """A dataset that always returns the same fixed chunk."""
    
    def __init__(self, audio, sensor, touch, touch_raw, sound_id, sound, curve, offset, note):
        self.audio = audio
        self.sensor = sensor
        self.touch = touch
        self.touch_raw = touch_raw
        self.sound_id = sound_id
        self.sound = sound
        self.curve = curve
        self.offset = offset
        self.note = note
        self.chunk_samples = len(audio)
    
    def __len__(self):
        # Return a large number so the DataLoader doesn't run out
        return 10000
    
    def __getitem__(self, idx):
        return {
            "audio": self.audio.astype(np.float32),
            "sensor": self.sensor.astype(np.float32),
            "touch": self.touch.astype(np.float32),
            "touch_raw": self.touch_raw.astype(np.float32),
            "sound_id": self.sound_id,
            "sound": self.sound,
            "curve": self.curve,
            "offset": self.offset,
            "note": self.note.astype(np.float32),
        }


def find_good_chunk(audio, touch_raw, min_touch_seconds=2.0, chunk_s=10.0, sr=48000, hop=480):
    """
    Find a chunk where touch_raw > 0 for at least min_touch_seconds.
    Returns (offset_frames, touch_seconds_in_chunk, chosen_chunk_audio, chosen_chunk_sensor, chosen_chunk_touch).
    """
    chunk_frames = int(chunk_s * 100)  # 100 Hz frame rate
    hop_frames = hop  # 480 samples = 1 frame
    
    # Count frames where touch_raw > 0 in each possible chunk
    touch_frames = (touch_raw > 0).astype(np.float32)
    
    max_good_frames = 0
    best_offset = 0
    
    for offset in range(0, len(touch_raw) - chunk_frames + 1, hop_frames):
        chunk_touch = touch_frames[offset:offset + chunk_frames]
        good_frames = chunk_touch.sum()
        if good_frames > max_good_frames:
            max_good_frames = good_frames
            best_offset = offset
    
    touch_seconds = max_good_frames / 100.0  # 100 Hz
    
    # Extract the chunk
    offset = int(best_offset)
    end_frame = offset + chunk_frames
    end_sample = end_frame * 480
    
    chunk_audio = audio[480*offset : 480*end_frame]
    chunk_sensor = touch_raw[offset:end_frame]  # Use touch_raw as sensor
    _t = touch_raw[offset:end_frame]
    # v5: 3 touch features - log1p(t), exp(-t/0.5), clip(t,0,40)
    chunk_touch = np.stack([np.log1p(_t), np.exp(-_t / 0.5), np.clip(_t, 0.0, 40.0)], axis=-1)
    
    # Pad if needed
    if len(chunk_audio) < chunk_s * 100 * 480:
        pad_samples = int(chunk_s * 100 * 480) - len(chunk_audio)
        chunk_audio = np.pad(chunk_audio, (0, pad_samples), mode="constant")
        pad_frames = pad_samples // 480
        chunk_sensor = np.pad(chunk_sensor, (0, pad_frames), mode="constant")
        chunk_touch = np.pad(chunk_touch, (0, pad_frames), mode="constant")
    
    return best_offset, touch_seconds, chunk_audio, chunk_sensor, chunk_touch


def evaluate_model(model, val_items, chunk_s, scales, overlap, device):
    """Evaluate model on validation items and return loss and per-sound losses."""
    model.eval()
    val_losses = []
    silence_losses = []
    sound_names_ordered = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
    val_losses_by_sound = {s: [] for s in sound_names_ordered}
    silence_by_sound = {s: [] for s in sound_names_ordered}
    
    with torch.no_grad():
        for val_item in val_items:
            # Prepare batch
            audio = torch.from_numpy(val_item["audio"][:int(chunk_s * 100 * 480)]).unsqueeze(0).to(device)
            sensor = torch.from_numpy(val_item["sensor"][:int(chunk_s * 100)]).unsqueeze(0).unsqueeze(-1).to(device)
            touch = torch.from_numpy(val_item["touch"][:int(chunk_s * 100)]).unsqueeze(0).to(device)
            if touch.dim() == 2:
                touch = touch.unsqueeze(-1)
            sound_id = torch.tensor([val_item["sound_id"]], dtype=torch.long, device=device)
            note = torch.from_numpy(np.asarray(val_item.get("note", np.zeros_like(val_item["sensor"])), dtype=np.float32)[:int(chunk_s * 100)]).unsqueeze(0).to(device)
            if note.dim() == 2:
                note = note.unsqueeze(-1)

            pred = model(sensor, touch, sound_id, note=note).squeeze(-1)
            val_l = model.loss_fn(pred, audio, scales, overlap)
            val_losses.append(val_l.item())
            
            # Silence baseline
            sil_l = model.silence_loss(audio, scales, overlap)
            silence_losses.append(sil_l.item())
            
            # Collect by sound
            sound = val_item["sound"]
            if sound in val_losses_by_sound:
                val_losses_by_sound[sound].append(val_l.item())
            if sound in silence_by_sound:
                silence_by_sound[sound].append(sil_l.item())
    
    model.train()
    
    # Compute means
    val_loss = np.mean(val_losses) if val_losses else None
    silence_loss = np.mean(silence_losses) if silence_losses else None
    
    return val_loss, silence_loss, val_losses_by_sound, silence_by_sound


def prepare_batch(item, device, chunk_s):
    """Prepare a validation batch."""
    audio = torch.from_numpy(item["audio"][:int(chunk_s * 100 * 480)]).unsqueeze(0).to(device)
    sensor = torch.from_numpy(item["sensor"][:int(chunk_s * 100)]).unsqueeze(0).unsqueeze(-1).to(device)
    touch = torch.from_numpy(item["touch"][:int(chunk_s * 100)]).unsqueeze(0).to(device)
    if touch.dim() == 2:
        touch = touch.unsqueeze(-1)
    sound_id = torch.tensor([item["sound_id"]], dtype=torch.long, device=device)
    note = torch.from_numpy(np.asarray(item.get("note", np.zeros_like(item["sensor"])), dtype=np.float32)[:int(chunk_s * 100)]).unsqueeze(0).to(device)
    if note.dim() == 2:
        note = note.unsqueeze(-1)
    return {
        "audio": audio,
        "sensor": sensor,
        "touch": touch,
        "note": note,
        "sound_id": sound_id,
        "sound": item["sound"],
        "curve": item["curve"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_sensor.yaml")
    p.add_argument("--name", default="sensor_debug")
    p.add_argument("--root", default="runs/bakeoff_ddsp_poc")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--no_restart", action="store_true")
    p.add_argument("--overfit_sound", default="03_harmonicSwarm")
    p.add_argument("--overfit_curve", default="001")
    args = p.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    # Build f0_base list in order of sound_ids from index.json
    with open("bakeoff_data_aligned/index.json") as f:
        index = json.load(f)
    
    with open("bakeoff_data_aligned/f0_base.json") as f:
        f0_base_json = json.load(f)
    
    # Order by sound_ids integer values
    sound_names = sorted(f0_base_json.keys(), key=lambda k: index["sound_ids"][k])
    f0_base = [f0_base_json[s]["f0_hz"] for s in sound_names]
    override = config["model"].get("f0_base_hz_override")
    if override:
        f0_base = list(override)  # SC-source roots, not pYIN
    print(f" f0_base Hz: {f0_base}")

    # Run directory
    run_dir = os.path.join(args.root, args.name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config with additional info
    run_config = config.copy()
    run_config["seed"] = args.seed
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd="/home/marcello/Documents/interactive_ddsp"
        ).decode().strip()
    except:
        git_hash = "unknown"
    run_config["git_hash"] = git_hash
    run_config["cli_args"] = vars(args)
    
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f)
    
    # Logging
    writer = SummaryWriter(run_dir, flush_secs=20)
    
    # CSV header with per-sound val columns
    sound_names_ordered = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
    csv_header = "step,train_loss,report_loss,val_loss,silence_loss," + ",".join([f"val_{s}" for s in sound_names_ordered]) + "," + ",".join([f"sil_{s}" for s in sound_names_ordered]) + "\n"
    log_file = open(os.path.join(run_dir, "train_log.csv"), "w")
    log_file.write(csv_header)
    
    # Training loop (may restart up to 3 times)
    max_restarts = 3
    for restart_num in range(max_restarts):
        print(f"\n=== Training restart #{restart_num} ===")
        
        # Reset seeds for restart
        torch.manual_seed(args.seed + restart_num)
        np.random.seed(args.seed + restart_num)
        random.seed(args.seed + restart_num)
        
        # Create model
        model_config = config["model"].copy()
        model_config.pop("f0_base_hz_override", None)
        # Remove LFO base parameters (they're registered as buffers in the model)
        model_config.pop("lfo_rate_base", None)
        model_config.pop("lfo_depth_mul_base", None)
        model_config.pop("lfo_depth_add_base", None)
        model_config["f0_base_hz"] = f0_base
        model_config["tilt_init"] = True  # Always use tilted init
        model = SensorDDSP(**model_config).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Create datasets
        root = config["data"]["root"]
        chunk_s = config["data"]["chunk_s"]
        
        # Training dataset
        train_dataset = AlignedChunkDataset(
            root, split="train", chunk_s=chunk_s, 
            sounds=config["data"]["train_sounds"],
            curves=config["data"]["train_curves"],
            epoch_len=None, seed=args.seed + restart_num, train=True
        )
        
        # Validation dataset
        val_dataset = AlignedChunkDataset(
            root, split="val", chunk_s=chunk_s,
            sounds=config["data"]["val_sounds"],
            curves=config["data"]["val_curves"],
            epoch_len=None, seed=args.seed + restart_num, train=False
        )

        # Overfit mode: use one fixed file
        if args.overfit:
            # Find a good chunk from overfit_sound, overfit_curve
            test_item = train_dataset.full(args.overfit_sound, args.overfit_curve)
            audio = test_item["audio"]
            touch_raw = test_item["touch_raw"]
            
            offset, touch_seconds, chunk_audio, chunk_sensor, chunk_touch = find_good_chunk(
                audio, touch_raw, 
                min_touch_seconds=2.0, 
                chunk_s=chunk_s
            )
            print(f"Overfit: using file {args.overfit_sound}_{args.overfit_curve}, offset {offset} frames ({offset/100:.2f}s), "
                  f"touch > 0 for {touch_seconds:.2f}s")
            
            # Create FixedChunkDataset that always returns the same chunk
            fixed_dataset = FixedChunkDataset(
                audio=chunk_audio,
                sensor=chunk_sensor,
                touch=chunk_touch,
                touch_raw=test_item["touch_raw"][offset:offset+int(chunk_s*100)],
                sound_id=test_item["sound_id"],
                sound=test_item["sound"],
                curve=test_item["curve"],
                offset=offset,
                note=np.asarray(test_item.get("note", np.zeros(len(chunk_sensor))), dtype=np.float32)[offset:offset + len(chunk_sensor)]
            )
            
            train_loader = torch.utils.data.DataLoader(
                fixed_dataset, batch_size=min(args.batch, 2), shuffle=False, drop_last=True
            )
            
            # Store the fixed chunk for later evaluation
            fixed_chunk = {
                "audio": chunk_audio,
                "sensor": chunk_sensor,
                "touch": chunk_touch,
                "note": np.asarray(test_item.get("note", np.zeros(len(chunk_sensor))), dtype=np.float32)[offset:offset + len(chunk_sensor)],
                "sound_id": test_item["sound_id"],
            }
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch, shuffle=True, drop_last=True
            )
            fixed_chunk = None

        # Validation: use ALL non-overlapping chunks from val_dataset
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=8, shuffle=False, drop_last=False
        )
        
        # Prepare val data for evaluation - collect all items
        val_items = []
        sound_curve_pairs = []
        for sound in ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]:
            for curve in ["017", "018", "019", "020"]:
                sound_curve_pairs.append((sound, curve))
        
        for sound, curve in sound_curve_pairs:
            item = val_dataset.full(sound, curve)
            val_items.append(item)
        
        # Optimizer
        lr_start = config["train"]["lr_start"]
        lr_stop = config["train"]["lr_stop"]
        decay_over = config["train"].get("decay_over", args.steps)
        
        # For overfit mode, use lr_start directly (3e-4 based on leader's diag scripts)
        if args.overfit:
            lr_start = 3e-4
            decay_over = 1000  # Don't decay during overfit
        
        opt = torch.optim.Adam(model.parameters(), lr=lr_start)
        step = 0
        timing_window = 25
        start_time = None
        step_count = 0

        # Gradient clipping
        max_grad_norm = 5.0

        # LR scheduler: warm-up then constant (based on diag_curr.py)
        def lr_at(step):
            # Warm-up for first 100 steps: min(1.0, (step+1)/100)
            warmup = min(1.0, (step + 1) / 100.0)
            return lr_start * warmup

        # Training loop
        step = 0
        best_val_loss = float("inf")
        collapse_detected = False
        rms_history = []  # Track last 10 RMS ratios for moving average check
        
        scales = config["train"]["scales"]
        overlap = config["train"]["overlap"]
        
        print(f"\nStarting training for {args.steps} steps")
        print(f"Train dataset: {len(train_dataset)} items")
        print(f"Val dataset: {len(val_dataset)} items")
        print(f"Val evaluation every {args.eval_every} steps")
        print()
        
        last_print_step = 0
        last_print_time = time.time()  # Track when we last printed for steps/sec
        step_times = []  # Track timestamps for steps in current 25-step window
        last_eval_step = -args.eval_every  # Ensure we eval at step 0
        
        # Initial evaluation at step 0 (before any updates)
        val_loss, silence_loss, val_losses_by_sound, silence_by_sound = evaluate_model(
            model, val_items, chunk_s, scales, overlap, device
        )
        best_val_loss = val_loss if val_loss is not None else float("inf")
        
        if val_loss is not None:
            print(f"Initial val_loss: {val_loss:.6f}, silence_loss: {silence_loss:.6f}")
            model.train()  # Reset to train mode after eval
        
        for epoch in range((args.steps // len(train_loader)) + 2):
            for batch in train_loader:
                if step >= args.steps:
                    break
                    
                # Update LR
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                
                # Move to device
                audio = batch["audio"].to(device)
                sensor = batch["sensor"].to(device).unsqueeze(-1)
                touch = batch["touch"].to(device)
                if touch.dim() == 2:
                    touch = touch.unsqueeze(-1)  # (B,T,1); dataset may return (B,T,2)
                sound_id = batch["sound_id"].to(device)
                note = batch["note"].to(device)
                if note.dim() == 2:
                    note = note.unsqueeze(-1)
                
                # Forward
                pred = model(sensor, touch, sound_id, note=note).squeeze(-1)  # (B, L)
                
                # Compute curriculum loss
                loss = model.loss_curriculum(pred, audio, scales, overlap, step, touch=touch, sound_id=sound_id)
                
                # Compute report loss (comparable loss_fn)
                report_loss = model.loss_fn(pred, audio, scales, overlap)
                
                # Guard: skip batch if loss is not finite (belt-and-braces against NaN)
                if not torch.isfinite(loss).item():
                    print(f"  Skipping batch at step {step}: loss={loss.item():.6e}")
                    opt.zero_grad()  # Clear gradients
                    continue
                
                # Backward
                opt.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                opt.step()
                
                # Check for collapse every 25 steps starting at step 400 (not 300, to avoid false positives during ramp)
                # Also check at evaluation points
                if step >= 400 and step % args.eval_every == 0:
                    pred_rms = pred.pow(2).mean().sqrt().item()
                    target_rms = audio.pow(2).mean().sqrt().item()
                    rms_ratio = pred_rms / (target_rms + 1e-8)
                    
                    # Track RMS history for moving average check
                    rms_history.append(rms_ratio)
                    if len(rms_history) > 10:
                        rms_history.pop(0)
                    
                    # Check for collapse: current RMS ratio or moving average too low
                    current_collapse = rms_ratio < 0.1
                    avg_rms = sum(rms_history) / len(rms_history) if rms_history else 0
                    avg_collapse = avg_rms < 0.1
                    
                    if current_collapse or avg_collapse:
                        collapse_detected = True
                        print(f"  COLLAPSE DETECTED at step {step}: RMS ratio {rms_ratio:.6f}, avg {avg_rms:.6f}")
                        
                        if not args.no_restart and restart_num < max_restarts - 1:
                            print(f"  Restarting training with seed {args.seed + restart_num + 1}")
                            # Break out of all loops and restart
                            break
                        else:
                            print(f"  Continuing with --no_restart or max restarts reached")
                
                # Track timestamp after each step for timing measurements
                step_times.append(time.time())
                
                # Logging - print every 25 steps with steps/sec
                # Always evaluate at step 0, eval_every, and last step
                val_loss = None
                silence_loss = None
                val_losses_by_sound = {}
                
                should_eval = (step == 0) or (step % args.eval_every == 0) or (step == args.steps - 1)
                
                if should_eval:
                    val_loss, silence_loss, val_losses_by_sound, silence_by_sound = evaluate_model(
                        model, val_items, chunk_s, scales, overlap, device
                    )
                    
                    # Save best model after evaluation
                    if val_loss is not None and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), os.path.join(run_dir, "state_best.pth"))
                        print(f"  New best val_loss: {val_loss:.6f}")
                
                # Also print at the last step (even if not 25 steps from last)
                should_print = (step - last_print_step >= 25) or (step == args.steps - 1)
                
                if should_print and len(step_times) >= 2:
                    # Calculate time elapsed since last_print_step
                    # step_times[-1] is time after current step, step_times[0] is time after last_print_step
                    # But we want to count steps from last_print_step to step (inclusive)
                    # So we measure from step_times[last_print_step % 25] to step_times[-1]
                    
                    # Find the index of the first timestamp in this 25-step window
                    window_start_idx = max(0, len(step_times) - 26)  # Include 25 steps: 26 timestamps
                    step_start_time = step_times[window_start_idx] if window_start_idx < len(step_times) else step_times[0]
                    step_end_time = step_times[-1]
                    
                    # Use sync for accurate CUDA timing
                    torch.cuda.synchronize()
                    step_time = step_end_time - step_start_time
                    steps_sec = 25.0 / step_time if step_time > 0 else 0
                    print(f"Steps/sec: {steps_sec:.2f}")

                    # Log to CSV
                    val_loss_str = f"{val_loss:.6f}" if val_loss is not None else ""
                    silence_loss_str = f"{silence_loss:.6f}" if silence_loss is not None else ""
                    val_by_sound_str = ",".join([
                        f"{np.mean(val_losses_by_sound.get(s, [])):.6f}" if val_losses_by_sound.get(s, []) else ""
                        for s in sound_names_ordered
                    ])
                    sil_by_sound_str = ",".join([
                        f"{np.mean(silence_by_sound.get(s, [])):.6f}" if silence_by_sound.get(s, []) else ""
                        for s in sound_names_ordered
                    ])
                    log_file.write(f"{step},{loss.item():.6f},{report_loss.item():.6f},{val_loss_str},{silence_loss_str},{val_by_sound_str},{sil_by_sound_str}\n")
                    log_file.flush()
                    
                    # If we're also at a print step, print the val loss
                    if should_eval:
                        print(f"  Val by sound:")
                        for s in sound_names_ordered:
                            if val_losses_by_sound.get(s, []):
                                print(f"    {s}: val_loss={np.mean(val_losses_by_sound[s]):.6f}, silence_loss={np.mean(silence_by_sound.get(s, [])):.6f}")
                    
                    if val_loss is not None:
                        print(f"step {step:5d}  train_loss {loss.item():.6f}  report_loss {report_loss.item():.6f}  val_loss {val_loss:.6f}  silence_loss {silence_loss:.6f}  lr {lr_at(step):.2e}  steps/sec {steps_sec:.2f}")
                    else:
                        print(f"step {step:5d}  train_loss {loss.item():.6f}  report_loss {report_loss.item():.6f}  lr {lr_at(step):.2e}  steps/sec {steps_sec:.2f}")
                    
                    last_print_step = step
                    # Clear step_times for next window (keep only what we need for overlap)
                    step_times = []
                
                # TensorBoard logging
                writer.add_scalar("loss/train", loss.item(), step)
                writer.add_scalar("loss/report", report_loss.item(), step)
                writer.add_scalar("lr", lr_at(step), step)
                
                # Reverb params
                if step >= 400 and step % args.eval_every == 0:
                    writer.add_scalar("reverb_decay", model.reverb.decay.item(), step)
                    writer.add_scalar("reverb_wet", model.reverb.wet.item(), step)
                
                step += 1
            
            if step >= args.steps:
                break
            
            if collapse_detected and not args.no_restart and restart_num < max_restarts - 1:
                break  # Restart
        
        if step >= args.steps:
            break
        
        if collapse_detected and not args.no_restart and restart_num < max_restarts - 1:
            print(f"  Restarting training with seed {args.seed + restart_num + 1}")
            # Reset optimizer, step, and last_print_step before restarting
            opt = torch.optim.Adam(model.parameters(), lr=lr_start)
            step = 0
            timing_window = 25
            start_time = None
            step_count = 0
            last_print_step = 0
            continue  # Restart
    
    # End of restart loop - final evaluation runs only on normal completion (not on restart)
    if step >= args.steps:
        # Final evaluation at last step (step 60)
        if step != args.steps - 1:
            # Eval at last step
            val_loss, silence_loss, val_losses_by_sound, silence_by_sound = evaluate_model(
                model, val_items, chunk_s, scales, overlap, device
            )
            
            if val_loss is not None:
                print(f"  Val by sound:")
                for s in sound_names_ordered:
                    if val_losses_by_sound.get(s, []):
                        print(f"    {s}: val_loss={np.mean(val_losses_by_sound[s]):.6f}, silence_loss={np.mean(silence_by_sound.get(s, [])):.6f}")
        
        # Final evaluation for final model
        model.eval()
        val_losses = []
        silence_losses = []
        val_losses_by_sound = {s: [] for s in sound_names_ordered}
        silence_by_sound = {s: [] for s in sound_names_ordered}
        with torch.no_grad():
            for val_item in val_items:
                val_batch = prepare_batch(val_item, device, chunk_s)
                val_pred = model(
                    val_batch["sensor"],
                    val_batch["touch"],
                    val_batch["sound_id"],
                    note=val_batch.get("note")
                ).squeeze(-1)
                val_l = model.loss_fn(val_pred, val_batch["audio"], scales, overlap)
                sil_l = model.silence_loss(val_batch["audio"], scales, overlap)
                
                val_losses.append(val_l.item())
                silence_losses.append(sil_l.item())
                
                sound = val_batch["sound"]
                if sound in val_losses_by_sound:
                    val_losses_by_sound[sound].append(val_l.item())
                if sound in silence_by_sound:
                    silence_by_sound[sound].append(sil_l.item())
        
        final_val_loss = np.mean(val_losses)
        final_silence_loss = np.mean(silence_losses)
        
        # Log final
        val_loss_str = f"{final_val_loss:.6f}"
        silence_loss_str = f"{final_silence_loss:.6f}"
        val_by_sound_str = ",".join([
            f"{np.mean(val_losses_by_sound.get(s, [])):.6f}" if val_losses_by_sound.get(s, []) else ""
            for s in sound_names_ordered
        ])
        sil_by_sound_str = ",".join([
            f"{np.mean(silence_by_sound.get(s, [])):.6f}" if silence_by_sound.get(s, []) else ""
            for s in sound_names_ordered
        ])
        log_file.write(f"{step},{loss.item():.6f},{report_loss.item():.6f},{val_loss_str},{silence_loss_str},{val_by_sound_str},{sil_by_sound_str}\n")
        
        print(f"\nTraining complete!")
        print(f"  Final train loss: {loss.item():.6f}")
        print(f"  Final report loss: {report_loss.item():.6f}")
        print(f"  Final val loss: {final_val_loss:.6f}")
        print(f"  Final silence loss: {final_silence_loss:.6f}")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"  Run directory: {run_dir}")
        
        # Save final model before closing log
        torch.save(model.state_dict(), os.path.join(run_dir, "state_final.pth"))

# After the outer for restart_num ... loop and its if step >= args.steps: break / continue handling
        
        # Close log file and tensorboard
        log_file.close()
        writer.close()
        
        # Overfit pass/fail check
        if args.overfit:
            # Evaluate on fixed chunk in eval mode
            model.eval()
            with torch.no_grad():
                audio_t = torch.from_numpy(fixed_chunk["audio"]).unsqueeze(0).to(device)
                sensor_t = torch.from_numpy(fixed_chunk["sensor"]).unsqueeze(0).unsqueeze(-1).to(device)
                touch_t = torch.from_numpy(fixed_chunk["touch"]).unsqueeze(0).to(device)
                if touch_t.dim() == 2:
                    touch_t = touch_t.unsqueeze(-1)
                sound_id_t = torch.tensor([fixed_chunk["sound_id"]], dtype=torch.long, device=device)
                note_t = torch.from_numpy(np.asarray(fixed_chunk["note"], dtype=np.float32)).unsqueeze(0).to(device)
                if note_t.dim() == 2:
                    note_t = note_t.unsqueeze(-1)
                
                pred = model(sensor_t, touch_t, sound_id_t, note=note_t).squeeze(0).squeeze(-1)
                
                # Loss
                final_loss = model.loss_fn(pred, audio_t, scales, overlap).item()
                silence_loss = model.silence_loss(audio_t, scales, overlap).item()
                
                # RMS ratio
                pred_rms = pred.pow(2).mean().sqrt().item()
                target_rms = audio_t.pow(2).mean().sqrt().item()
                rms_ratio = pred_rms / (target_rms + 1e-8)
                
                # Peak
                pred_peak = pred.abs().max().item()

                # Sensor ablation: zeroed and shuffled (rolled) sensor must change output
                from scipy.stats import pearsonr
                sensor_zero = torch.zeros_like(sensor_t)
                pred_zero = model(sensor_zero, touch_t, sound_id_t, note=note_t).squeeze(0).squeeze(-1)
                sensor_shuf = torch.roll(sensor_t.clone(), shifts=sensor_t.shape[1] // 2, dims=1)
                pred_shuf = model(sensor_shuf, touch_t, sound_id_t, note=note_t).squeeze(0).squeeze(-1)

                def _rms_envelope(x, frame_size=480, hop=480):
                    x = x.detach().cpu().numpy().ravel()
                    n = max((len(x) - frame_size) // hop + 1, 1)
                    e = np.array([np.sqrt(np.mean(x[i*hop:i*hop+frame_size]**2) + 1e-8) for i in range(n)])
                    return np.log(e + 1e-4)

                rms_corr_zero, _ = pearsonr(_rms_envelope(pred.squeeze()), _rms_envelope(pred_zero))
                rms_corr_shuf, _ = pearsonr(_rms_envelope(pred.squeeze()), _rms_envelope(pred_shuf))

                # Spectral centroid envelope correlation (timbre check)
                def _spectral_centroid(x, sr=48000, frame_size=2048, hop=480):
                    x = np.asarray(x).ravel()
                    n = max((len(x) - frame_size) // hop + 1, 1)
                    win = np.hanning(frame_size)
                    freqs = np.fft.rfftfreq(frame_size, 1 / sr)
                    out = np.zeros(n)
                    for i in range(n):
                        spec = np.abs(np.fft.rfft(x[i*hop:i*hop+frame_size] * win))
                        out[i] = np.sum(spec * freqs) / (np.sum(spec) + 1e-8)
                    return out

                centroid_corr, _ = pearsonr(_spectral_centroid(pred.squeeze().cpu().numpy()),
                                            _spectral_centroid(audio_t.squeeze().cpu().numpy()))

            # Pass criteria:
            # (i) final_loss < 0.6 * silence_loss
            # (ii) rms_ratio in [0.4, 2.0]
            # (iii) pred_peak > 1e-3
            # (iv) shuffled-sensor RMS envelope corr < 0.5 (model is sensor-conditioned)
            # (v) spectral centroid corr > 0.6 (timbre matches)
            cond1 = final_loss < 0.6 * silence_loss
            cond2 = 0.4 <= rms_ratio <= 2.0
            cond3 = pred_peak > 1e-3
            cond4 = bool(rms_corr_shuf < 0.5)
            cond5 = bool(centroid_corr > 0.6)
            passed = cond1 and cond2 and cond3 and cond4 and cond5

            print(f"\nOverfit test:")
            print(f"  Chunk offset: {offset} frames ({offset/100:.2f}s)")
            print(f"  Touch seconds in chunk: {touch_seconds:.2f}s")
            print(f"  Silence loss: {silence_loss:.6f}")
            print(f"  Final loss: {final_loss:.6f}")
            print(f"  RMS ratio: {rms_ratio:.6f}")
            print(f"  Peak: {pred_peak:.6e}")
            print(f"  RMS corr zeroed sensor: {rms_corr_zero:.4f}")
            print(f"  RMS corr shuffled sensor: {rms_corr_shuf:.4f}")
            print(f"  Spectral centroid corr: {centroid_corr:.4f}")
            print(f"  PASS criteria: final_loss < 0.6 * silence_loss = {silence_loss * 0.6:.6f}: {cond1}")
            print(f"               0.4 <= rms_ratio <= 2.0: {cond2}")
            print(f"               peak > 1e-3: {cond3}")
            print(f"               shuffled-sensor corr < 0.5: {cond4}")
            print(f"               centroid corr > 0.6: {cond5}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")

            # Per-run results artefact
            results = {
                "run_name": args.name,
                "sound": args.overfit_sound,
                "curve": args.overfit_curve,
                "seed": args.seed,
                "steps": args.steps,
                "silence_loss": silence_loss,
                "final_loss": final_loss,
                "rms_ratio": rms_ratio,
                "pred_peak": pred_peak,
                "rms_corr_zero": float(rms_corr_zero),
                "rms_corr_shuffled": float(rms_corr_shuf),
                "centroid_corr": float(centroid_corr),
                "restarts": restart_num,
                "passed": bool(passed),
                "pass_criteria": {
                    "loss": bool(cond1),
                    "rms_ratio": bool(cond2),
                    "peak": bool(cond3),
                    "ablation": bool(cond4),
                    "spectral": bool(cond5),
                },
            }
            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            # Aggregate results table
            global_results_path = os.path.join(args.root, "RESULTS.json")
            global_results = []
            if os.path.exists(global_results_path):
                with open(global_results_path, "r") as f:
                    global_results = json.load(f)
            global_results.append(results)
            tmp_path = global_results_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(global_results, f, indent=2)
            os.replace(tmp_path, global_results_path)

if __name__ == "__main__":
    main()
