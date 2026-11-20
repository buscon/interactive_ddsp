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


def find_good_chunk(audio, touch_raw, min_touch_seconds=2.0, chunk_s=10.0, sr=48000, hop=480):
    """
    Find a chunk where touch_raw > 0 for at least min_touch_seconds.
    Returns (offset_frames, touch_seconds_in_chunk).
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
    return best_offset, touch_seconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_sensor.yaml")
    p.add_argument("--name", default="sensor_debug")
    p.add_argument("--root", default="runs/bakeoff_ddsp_poc")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
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
        
        offset, touch_seconds = find_good_chunk(
            audio, touch_raw, 
            min_touch_seconds=2.0, 
            chunk_s=chunk_s
        )
        print(f"Overfit: using file 03_harmonicSwarm_001, offset {offset} frames ({offset/100:.2f}s), "
              f"touch > 0 for {touch_seconds:.2f}s")
        
        # Load that specific chunk repeatedly
        train_dataset = AlignedChunkDataset(
            root, split="train", chunk_s=chunk_s,
            sounds=["03_harmonicSwarm"],
            curves=["001"],
            epoch_len=args.steps * args.batch,  # Enough for all steps
            seed=args.seed, train=True
        )
        # Force it to always return the same chunk by overriding __getitem__
        original_getitem = train_dataset.__getitem__
        def fixed_getitem(idx):
            # Return the chunk at the found offset
            return {
                "audio": audio[offset*480:(offset+int(chunk_s*100))*480].astype(np.float32),
                "sensor": test_item["sensor"][offset:offset+int(chunk_s*100)].astype(np.float32),
                "touch": test_item["touch"][offset:offset+int(chunk_s*100)].astype(np.float32),
                "touch_raw": test_item["touch_raw"][offset:offset+int(chunk_s*100)].astype(np.float32),
                "sound_id": test_item["sound_id"],
                "sound": test_item["sound"],
                "curve": test_item["curve"],
                "offset": offset,
            }
        train_dataset.__getitem__ = fixed_getitem
        train_dataset.epoch_len = args.steps * args.batch
        
        # Use batch 1 or 2 as specified
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=min(args.batch, 2), shuffle=False, drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, drop_last=True
        )

    # Validation subset: fixed 40 chunks covering all 5 sounds
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, drop_last=False
    )
    # Sample 40 items deterministically
    val_items = []
    sound_curve_pairs = []
    for sound in ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]:
        for curve in ["017", "018", "019", "020"]:
            sound_curve_pairs.append((sound, curve))
    
    for sound, curve in sound_curve_pairs[:40]:
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
    log_file = open(os.path.join(run_dir, "train_log.csv"), "w")
    log_file.write("step,train_loss,val_loss\n")
    
    # Training loop
    step = 0
    best_val_loss = float("inf")
    
    scales = config["train"]["scales"]
    overlap = config["train"]["overlap"]
    
    print(f"\nStarting training for {args.steps} steps")
    print(f"Train dataset: {len(train_dataset)} items")
    print(f"Val dataset: {len(val_dataset)} items")
    print(f"Val evaluation every 250 steps")
    print()
    
    initial_loss = None
    last_print_step = 0
    start_time = time.time()
    
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
            
            # Overfit: track initial loss
            if step == 0 and args.overfit:
                initial_loss = loss.item()
                print(f"Initial loss: {initial_loss:.6f}")
            
            # Logging - print every 25 steps with steps/sec
            # Track time at each step for proper measurement
            if not hasattr(main, 'step_start_times'):
                main.step_start_times = []
            
            main.step_start_times.append(time.time())
            
            # When we hit 25 steps since last print
            if step - last_print_step >= 25:
                # Get time at last_print_step
                if last_print_step >= 0 and last_print_step < len(main.step_start_times):
                    prev_time = main.step_start_times[last_print_step]
                else:
                    prev_time = start_time if hasattr(main, 'start_time') else time.time()
                
                # Current time
                curr_time = time.time()
                
                # Calculate elapsed time for 25 steps
                step_time_elapsed = curr_time - prev_time
                steps_trained = 25
                
                steps_sec = steps_trained / step_time_elapsed if step_time_elapsed > 0 else 0
                
                # Compute val loss every 250 steps
                val_loss = None
                if step % 250 == 0 or step == args.steps - 1:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for val_item in val_items[:40]:
                            val_batch = prepare_batch(val_item, device)
                            val_pred = model(
                                val_batch["sensor"], 
                                val_batch["touch"], 
                                val_batch["sound_id"]
                            ).squeeze(-1)
                            val_l = model.loss_fn(val_pred, val_batch["audio"], scales, overlap)
                            val_losses.append(val_l.item())
                    val_loss = np.mean(val_losses)
                    model.train()
                    
                    log_file.write(f"{step},{loss.item():.6f},{val_loss:.6f}\n")
                    log_file.flush()
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), os.path.join(run_dir, "state_best.pth"))
                        print(f"  New best val_loss: {val_loss:.6f}")
                
                if val_loss is not None:
                    print(f"step {step:5d}  train_loss {loss.item():.6f}  val_loss {val_loss:.6f}  "
                          f"lr {lr_at(step):.2e}  steps/sec {steps_sec:.2f}")
                else:
                    print(f"step {step:5d}  train_loss {loss.item():.6f}  lr {lr_at(step):.2e}  steps/sec {steps_sec:.2f}")
                
                last_print_step = step
            
            # TensorBoard logging
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("lr", lr_at(step), step)
            
            # Reverb params
            if step % 250 == 0:
                writer.add_scalar("reverb_decay", model.reverb.decay.item(), step)
                writer.add_scalar("reverb_wet", model.reverb.wet.item(), step)
            
            step += 1
        
        if step >= args.steps:
            break
    
    # Final evaluation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for val_item in val_items[:40]:
            val_batch = prepare_batch(val_item, device)
            val_pred = model(
                val_batch["sensor"], 
                val_batch["touch"], 
                val_batch["sound_id"]
            ).squeeze(-1)
            val_l = model.loss_fn(val_pred, val_batch["audio"], scales, overlap)
            val_losses.append(val_l.item())
    final_val_loss = np.mean(val_losses)
    
    log_file.write(f"{step},{loss.item():.6f},{final_val_loss:.6f}\n")
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
        final_loss = loss.item()
        threshold = 0.6 * initial_loss
        passed = final_loss <= threshold
        
        print(f"\nOverfit test:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Target (0.6 * initial): {threshold:.6f}")
        print(f"  Result: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
