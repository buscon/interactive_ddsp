# V9 Attack-Lag Fix: Results

**Date:** 2026-11-24
**Branch:** `orch/v9lagfix` (from `orch/v9data-fix` @ `d0bd9d2`)
**Verdict:** **helps** by the criteria fixed in advance, but only partly. Nothing passes the
strict check yet.

## Change

`attack_lag_s`: [0.38, 0.38, 0.42, 0.39] → **[0.023, 0.026, 0.065, 0.025]** s, in the four
4-sound v9 configs and in the model's default buffer. The 0.4 s values came from the v8 dataset
(server latency). The new values are the v6 touch-layer onset plus the 21.3 ms limiter delay,
from `V9_LAG_MEASUREMENT.md`.

Training repeated the baseline exactly:

```
python3 train_sensor_ddsp_v5.py --config config_sensor_v9_stageB_fixed.yaml \
  --name v9_stageB_lagfix_s{0,1} --root $R --steps 2000 --seed {0,1}
python3 eval_stageB_v5.py --run $R/<run> --ckpt state_final.pth --out $R/<eval> --split val
```

`R = ~/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed`.

Provenance was checked by the leader and by Sonnet. The only config differences between
baseline and new runs are `name`, `git_hash` and `attack_lag_s`, and each checkpoint's lag
buffer matches its config. Every eval log names the right checkpoint. There is no NaN or error
in the training logs.

## Results (val split, 76 onsets per run)

| Run | lag | dpk (dB) | dtpk (ms) | \|dtpk\| (ms) | p_bump (dB) |
|---|---|---|---|---|---|
| `fixed_s0` | 0.4 s | −10.86 | 155 | 160 | 1.41 |
| `fixed_s1` | 0.4 s | −12.92 | 148 | 148 | 2.95 |
| `lagfix_s0` | v6 | **−8.14** | **97** | **97** | 2.41 |
| `lagfix_s1` | v6 | **−8.82** | **128** | **128** | 2.70 |
| **mean, old → new** | | −11.89 → **−8.48** (+3.4) | | 154 → **112.5** (−41.5) | 2.18 → 2.56 |

**Criteria fixed in advance:** mean |dtpk| improves by at least 30 ms (it improved by 41.5 ms),
and dpk is no worse by more than 1 dB (it improved by 3.4 dB). → **helps**.
Strict pass (|dpk| ≤ 4 dB and |dtpk| ≤ 20 ms): **0/4 sounds** in every run.

**Per seed pair:** all 4 new-vs-old pairs improve on both measures. On timing the gain is
−20 to −63 ms, so one pair misses the 30 ms bar. On level the gain is +2.0 to +4.8 dB.
The two new seeds are 31 ms apart.

### Per sound (2-seed mean |dtpk|, ms)

| Sound | old lag | new lag | Seed pairs | dpk old → new (dB) |
|---|---|---|---|---|
| canopy | 155.0 | 129.8 | 3 of 4 better | −10.6 → −8.2 |
| forestPad | 172.8 | **221.8 (worse)** | 4 of 4 worse | −6.2 → −6.3 |
| harmonicSwarm | 151.0 | 112.5 | 4 of 4 better | −14.9 → −12.0 |
| moss_and_mist | 128.8 | **64.5** | 4 of 4 better | −11.9 → −8.5 |

### The attack path is now used

Sonnet measured how much `fast_head` (the touch-gated attack path) adds. It compared the
0-300 ms onset peak with and without `fast_head` zeroed at inference:

| Run | canopy | forestPad | swarm | moss |
|---|---|---|---|---|
| `fixed_s0` | −0.09 | +0.03 | +0.01 | −0.13 |
| `fixed_s1` | −0.03 | −0.19 | +0.31 | +0.26 |
| `lagfix_s0` | **+3.08** | **+2.64** | **+3.43** | **+2.38** |
| `lagfix_s1` | +0.79 | +1.00 | +1.27 | +0.73 |

With the 0.4 s lag, the attack path did nothing (< 0.5 dB). With the v6 lag it adds 0.7-3.4 dB.
The difference between seeds here (2.9 vs 0.9 dB) roughly matches the difference in their attack
results.

### Dynamics

Spot-checked on 4 files: crest factor and envelope range. The envelope range is unchanged to
+3 dB. Crest factor is unchanged for seed 0 but drops by 1-4 dB in three seed-1 files
(forestPad −4.3 dB). Both old and new runs remain well below the target, by 5-10 dB in crest
and 25-30 dB in range. These are proxies, so listen to confirm.

## Caveats

- Only 2 seeds. The gain (−41.5 ms) is about as large as the spread between the new seeds (31 ms).
- forestPad timing got worse in every seed pair, from 173 to 222 ms. Its lag (26 ms) is close to
  canopy's, so the lag alone doesn't explain it. Not investigated.
- The attack is still about 112 ms late and 8.5 dB too quiet, so the lag is only part of the
  problem. See `V9_STATUS_AND_NEXT_STEPS.md` §3.1 for the earlier A4 comparison.
- `silence_loss` is identical across runs. It depends only on the target, so this is expected.

## Listening

`listening_v9_lagfix/` holds the seed-0 renders (16 val curves). Compare each curve with
`listening_v9_best/` (old lag).

## Suggested next steps

1. **Keep the v6 lag.** Run a third seed if the timing gain needs firmer support.
2. **forestPad:** look at its onsets. Compare the target touch-layer onset with the pred peak for
   curves 003, 007, 013 and 014.
3. **Make the attack path stronger:** it is now used but is still weak, and seed 1 barely uses
   it. Options: sound-conditioned `fast_head`, a higher `w_attack`, or the touch-layer
   auxiliary target (status doc §4 step 5).
4. The LFO-embedding freeze (step 2) and the pitch fix (step 3) from the status doc are still
   open.
