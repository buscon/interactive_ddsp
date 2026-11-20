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
sys.path.insert(0, "/home/marcello/Documents/interactive_ddsp")

from engine_ddsp.sensor_data import AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import multiscale_fft, safe_log


class FixedChunkDataset(torch.utils.data.Dataset):
    """A dataset that always returns the same fixed chunk."""
    
    def __init__(self, audio, sensor, touch, touch_raw, sound_id, sound, curve, offset):
        self.audio = audio
        self.sensor = sensor
        self.touch = touch
        self.touch_raw = touch_raw
        self.sound_id = sound_id
        self.sound = sound
        self.curve = curve
        self.offset = offset
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
        }


def evaluate_model(model, val_items, chunk_s, scales, overlap, device):
    """Evaluate model on validation items and return loss and per-sound losses."""
    model.eval()
    val_losses = []
    sound_names_ordered = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
    val_losses_by_sound = {s: [] for s in sound_names_ordered}
    
    with torch.no_grad():
        for val_item in val_items:
            # Prepare batch
            audio = torch.from_numpy(val_item["audio"][:int(chunk_s * 100 * 480)]).unsqueeze(0).to(device)
            sensor = torch.from_numpy(val_item["sensor"][:int(chunk_s * 100)]).unsqueeze(0).unsqueeze(-1).to(device)
            touch = torch.from_numpy(val_item["touch"][:int(chunk_s * 100)]).unsqueeze(0).unsqueeze(-1).to(device)
            sound_id = torch.tensor([val_item["sound_id"]], dtype=torch.long, device=device)
            
            pred = model(sensor, touch, sound_id).squeeze(-1)
            val_l = model.loss_fn(pred, audio, scales, overlap)
            val_losses.append(val_l.item())
            
            # Collect by sound
            sound = val_item["sound"]
            if sound in val_losses_by_sound:
                val_losses_by_sound[sound].append(val_l.item())
    
    model.train()
    
    if val_losses:
        return np.mean(val_losses), val_losses_by_sound
    return None, val_losses_by_sound


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
    chunk_touch = np.clip(touch_raw[offset:end_frame] / 30.0, 0.0, 2.0)
    
    # Pad if needed
    if len(chunk_audio) < chunk_s * 100 * 480:
        pad_samples = int(chunk_s * 100 * 480) - len(chunk_audio)
        chunk_audio = np.pad(chunk_audio, (0, pad_samples), mode="constant")
        pad_frames = pad_samples // 480
        chunk_sensor = np.pad(chunk_sensor, (0, pad_frames), mode="constant")
        chunk_touch = np.pad(chunk_touch, (0, pad_frames), mode="constant")
    
    return best_offset, touch_seconds, chunk_audio, chunk_sensor, chunk_touch


def compute_loss_on_fixed_chunk(model, audio, sensor, touch, sound_id, scales, overlap, device):
    """Compute loss on a fixed chunk in eval mode."""
    model.eval()
    with torch.no_grad():
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        sensor_t = torch.from_numpy(sensor).unsqueeze(0).unsqueeze(-1).to(device)
        touch_t = torch.from_numpy(touch).unsqueeze(0).unsqueeze(-1).to(device)
        sound_id_t = torch.tensor([sound_id], dtype=torch.long, device=device)
        
        pred = model(sensor_t, touch_t, sound_id_t).squeeze(-1)
        loss = model.loss_fn(pred, audio_t, scales, overlap)
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_sensor.yaml")
    p.add_argument("--name", default="sensor_debug")
    p.add_argument("--root", default="runs/bakeoff_ddsp_poc")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=250)
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
    print(f" f0_base Hz: {f0_base}")

    # Create model
    model_config = config["model"].copy()
    model_config["f0_base_hz"] = f0_base
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
        epoch_len=None, seed=args.seed, train=True
    )
    
    # Validation dataset
    val_dataset = AlignedChunkDataset(
        root, split="val", chunk_s=chunk_s,
        sounds=config["data"]["val_sounds"],
        curves=config["data"]["val_curves"],
        epoch_len=None, seed=args.seed, train=False
    )

    # Overfit mode: use one fixed file
    if args.overfit:
        # Find a good chunk from 03_harmonicSwarm, curve 001
        test_item = train_dataset.full("03_harmonicSwarm", "001")
        audio = test_item["audio"]
        touch_raw = test_item["touch_raw"]
        
        offset, touch_seconds, chunk_audio, chunk_sensor, chunk_touch = find_good_chunk(
            audio, touch_raw, 
            min_touch_seconds=2.0, 
            chunk_s=chunk_s
        )
        print(f"Overfit: using file 03_harmonicSwarm_001, offset {offset} frames ({offset/100:.2f}s), "
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
            offset=offset
        )
        
        train_loader = torch.utils.data.DataLoader(
            fixed_dataset, batch_size=min(args.batch, 2), shuffle=False, drop_last=True
        )
        
        # Store the fixed chunk for later evaluation
        fixed_chunk = {
            "audio": chunk_audio,
            "sensor": chunk_sensor,
            "touch": chunk_touch,
            "sound_id": test_item["sound_id"],
        }
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, drop_last=True
        )
        fixed_chunk = None

    # Validation: use ALL non-overlapping chunks from val_dataset
    # val_dataset already returns deterministic non-overlapping chunks with train=False
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
    
    # Prepare val data for evaluation
    def prepare_batch(item, device):
        audio = torch.from_numpy(item["audio"][:int(chunk_s * 100 * 480)]).unsqueeze(0).to(device)
        sensor = torch.from_numpy(item["sensor"][:int(chunk_s * 100)]).unsqueeze(0).unsqueeze(-1).to(device)
        touch = torch.from_numpy(item["touch"][:int(chunk_s * 100)]).unsqueeze(0).unsqueeze(-1).to(device)
        sound_id = torch.tensor([item["sound_id"]], dtype=torch.long, device=device)
        return {
            "audio": audio,
            "sensor": sensor,
            "touch": touch,
            "sound_id": sound_id,
            "sound": item["sound"],
            "curve": item["curve"],
        }

    # Optimizer
    lr_start = config["train"]["lr_start"]
    lr_stop = config["train"]["lr_stop"]
    decay_over = config["train"].get("decay_over", args.steps)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr_start)
    
    # Gradient clipping
    max_grad_norm = 5.0

    # LR scheduler: linear decay then constant
    def lr_at(step):
        if step >= decay_over:
            return lr_stop
        t = step / decay_over
        return lr_start * (1 - t) + lr_stop * t

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
    csv_header = "step,train_loss,val_loss," + ",".join([f"val_{s}" for s in sound_names_ordered]) + "\n"
    log_file = open(os.path.join(run_dir, "train_log.csv"), "w")
    log_file.write(csv_header)
    
    # Training loop
    step = 0
    best_val_loss = float("inf")
    
    scales = config["train"]["scales"]
    overlap = config["train"]["overlap"]
    
    print(f"\nStarting training for {args.steps} steps")
    print(f"Train dataset: {len(train_dataset)} items")
    print(f"Val dataset: {len(val_dataset)} items")
    print(f"Val evaluation every {args.eval_every} steps")
    print()
    
    initial_loss = None
    last_print_step = 0
    last_eval_step = -args.eval_every  # Ensure we eval at step 0
    start_time = time.time()
    
    # Initial evaluation at step 0 (before any updates)
    val_loss, val_losses_by_sound = evaluate_model(
        model, val_items, chunk_s, scales, overlap, device
    )
    best_val_loss = val_loss if val_loss is not None else float("inf")
    
    if val_loss is not None:
        print(f"Initial val_loss: {val_loss:.6f}")
    
    # Overfit: track initial loss on fixed chunk (model will be in train mode after this)
    initial_loss = None
    if args.overfit:
        initial_loss = compute_loss_on_fixed_chunk(
            model, fixed_chunk["audio"], fixed_chunk["sensor"], fixed_chunk["touch"],
            fixed_chunk["sound_id"], scales, overlap, device
        )
        print(f"Initial loss on fixed chunk: {initial_loss:.6f}")
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
            touch = batch["touch"].to(device).unsqueeze(-1)
            sound_id = batch["sound_id"].to(device)
            
            # Forward
            pred = model(sensor, touch, sound_id).squeeze(-1)  # (B, L)
            
            # Compute loss
            loss = model.loss_fn(pred, audio, scales, overlap)
            
            # Backward
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            opt.step()
            
            # Logging - print every 25 steps with steps/sec
            # Always evaluate at step 0, eval_every, and last step
            val_loss = None
            val_losses_by_sound = {}
            
            should_eval = (step == 0) or (step % args.eval_every == 0) or (step == args.steps - 1)
            
            if should_eval:
                val_loss, val_losses_by_sound = evaluate_model(
                    model, val_items, chunk_s, scales, overlap, device
                )
                
                # Save best model after evaluation
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(run_dir, "state_best.pth"))
                    print(f"  New best val_loss: {val_loss:.6f}")
                
                # If we're also at a print step, print the val loss
                should_print = (step - last_print_step >= 25) or (step == args.steps - 1)
                if should_print:
                    print(f"  Val by sound:")
                    for s in sound_names_ordered:
                        if val_losses_by_sound.get(s, []):
                            print(f"    {s}: {np.mean(val_losses_by_sound[s]):.6f}")
            
            should_print = (step - last_print_step >= 25) or (step == args.steps - 1)
            
            if should_print:
                # Measure time for 25 steps using wall clock
                torch.cuda.synchronize()
                step_start_time = time.time()
                
                # Calculate steps/sec for last 25 training steps
                torch.cuda.synchronize()
                step_time = time.time() - step_start_time
                steps_sec = 25.0 / step_time if step_time > 0 else 0
                
                # Log to CSV
                val_loss_str = f"{val_loss:.6f}" if val_loss is not None else ""
                val_by_sound_str = ",".join([
                    f"{np.mean(val_losses_by_sound.get(s, [])):.6f}" if val_losses_by_sound.get(s, []) else ""
                    for s in sound_names_ordered
                ])
                log_file.write(f"{step},{loss.item():.6f},{val_loss_str},{val_by_sound_str}\n")
                log_file.flush()
                
                # Print status
                if val_loss is not None:
                    print(f"step {step:5d}  train_loss {loss.item():.6f}  val_loss {val_loss:.6f}  "
                          f"lr {lr_at(step):.2e}  steps/sec {steps_sec:.2f}")
                    # Per-sound means
                    print(f"  Val by sound:")
                    for s in sound_names_ordered:
                        if val_losses_by_sound[s]:
                            print(f"    {s}: {np.mean(val_losses_by_sound[s]):.6f}")
                else:
                    print(f"step {step:5d}  train_loss {loss.item():.6f}  lr {lr_at(step):.2e}  steps/sec {steps_sec:.2f}")
                
                last_print_step = step
            
            # TensorBoard logging
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", lr_at(step), step)
            
            # Reverb params
            if step % args.eval_every == 0:
                writer.add_scalar("reverb_decay", model.reverb.decay.item(), step)
                writer.add_scalar("reverb_wet", model.reverb.wet.item(), step)
            
            step += 1
        
        if step >= args.steps:
            break
    
    # Final evaluation at last step (step 60)
    if step != args.steps - 1:
        # Eval at last step
        val_loss, val_losses_by_sound = evaluate_model(
            model, val_items, chunk_s, scales, overlap, device
        )
        
        if val_loss is not None:
            print(f"  Val by sound:")
            for s in sound_names_ordered:
                if val_losses_by_sound.get(s, []):
                    print(f"    {s}: {np.mean(val_losses_by_sound[s]):.6f}")
    
    # Final evaluation for final model
    model.eval()
    val_losses = []
    val_losses_by_sound = {s: [] for s in sound_names_ordered}
    with torch.no_grad():
        for val_item in val_items:
            val_batch = prepare_batch(val_item, device)
            val_pred = model(
                val_batch["sensor"], 
                val_batch["touch"], 
                val_batch["sound_id"]
            ).squeeze(-1)
            val_l = model.loss_fn(val_pred, val_batch["audio"], scales, overlap)
            val_losses.append(val_l.item())
            
            sound = val_batch["sound"]
            if sound in val_losses_by_sound:
                val_losses_by_sound[sound].append(val_l.item())
    
    final_val_loss = np.mean(val_losses)
    
    # Log final
    val_loss_str = f"{final_val_loss:.6f}"
    val_by_sound_str = ",".join([
        f"{np.mean(val_losses_by_sound.get(s, [])):.6f}" if val_losses_by_sound.get(s, []) else ""
        for s in sound_names_ordered
    ])
    log_file.write(f"{step},{loss.item():.6f},{val_loss_str},{val_by_sound_str}\n")
    log_file.close()
    writer.close()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(run_dir, "state_final.pth"))
    
    print(f"\nTraining complete!")
    print(f"  Final train loss: {loss.item():.6f}")
    print(f"  Final val loss: {final_val_loss:.6f}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Run directory: {run_dir}")
    
    # Overfit pass/fail check
    if args.overfit:
        final_loss = compute_loss_on_fixed_chunk(
            model, fixed_chunk["audio"], fixed_chunk["sensor"], fixed_chunk["touch"],
            fixed_chunk["sound_id"], scales, overlap, device
        )
        threshold = 0.6 * initial_loss
        passed = final_loss <= threshold
        
        print(f"\nOverfit test:")
        print(f"  Chunk offset: {offset} frames ({offset/100:.2f}s)")
        print(f"  Touch seconds in chunk: {touch_seconds:.2f}s")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Target (0.6 * initial): {threshold:.6f}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
