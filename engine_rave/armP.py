"""Arm P: sensor->latent mapper on frozen pretrained RAVE encoder/decoder (water_pondbrain).

Trains on curves 001-016 (train), evaluates on 017-020 (val) with Stage B metrics,
shared roll-2000 ablation, and Spearman control response.
"""
import os, sys, json, glob
import numpy as np
import torch
import torch.nn as nn

REPO = '/home/marcello/Documents/interactive_ddsp'
sys.path.insert(0, REPO)
from eval_stageB import compute_multiscale_loss

SR = 48000
SCALES = [4096, 2048, 1024, 512, 256, 128]
OVERLAP = 0.75
MODEL_TS = '/home/marcello/Documents/RAVE_SC_claude/models/water_pondbrain_b2048_r48000_z16.ts'
DATA = os.path.join(REPO, 'bakeoff_data_aligned')
OUT = os.path.join(os.path.dirname(__file__), '..', 'runs', 'rave_armP')
os.makedirs(OUT, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_files(split):
    idx = json.load(open(os.path.join(DATA, 'index.json')))
    sounds = sorted(idx['sound_ids'], key=lambda k: idx['sound_ids'][k])
    curves = idx['train_curves'] if split == 'train' else idx['val_curves']
    out = []
    for s in sounds:
        for c in curves:
            p = os.path.join(DATA, f'{s}_proximity_curve_{c}.npz')
            if os.path.exists(p):
                d = np.load(p)
                out.append(dict(sound=s, curve=c, sound_id=idx['sound_ids'][s],
                                audio=d['audio'].astype(np.float32) / 32768.0,
                                sensor=d['sensor'].astype(np.float32),
                                touch=d['touch_time'].astype(np.float32)))
    return out

def band_shape_corr(pred, tgt, bands=[(0,500),(500,2000),(2000,6000),(6000,12000),(12000,24000)]):
    from scipy.stats import pearsonr
    def mlb(x):
        x = np.asarray(x).ravel()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return np.log(np.array([np.sqrt(spec[(f>=lo)&(f<hi)].sum()+1e-12) for lo,hi in bands]) + 1e-6)
    return float(pearsonr(mlb(pred), mlb(tgt))[0])

def rms_env(x, frame=480, hop=480):
    x = np.asarray(x).ravel()
    n = max((len(x) - frame) // hop + 1, 1)
    return np.log(np.array([np.sqrt(np.mean(x[i*hop:i*hop+frame]**2)+1e-8) for i in range(n)]) + 1e-4)

def main():
    torch.manual_seed(0); np.random.seed(0)
    rave = torch.jit.load(MODEL_TS, map_location=device).eval()
    with torch.no_grad():  # measure latent hop
        z0 = rave.encode(torch.zeros(1, 1, 2*SR, device=device))
    hop = 2 * SR / z0.shape[-1]
    print(f'latent hop = {hop:.1f} samples ({SR/hop:.2f} Hz)')

    train = load_files('train'); val = load_files('val')
    print(f'{len(train)} train files, {len(val)} val files')

    # encode all files -> latents (cache)
    cache = os.path.join(OUT, 'latents.npz')
    lat = {}
    if os.path.exists(cache):
        lat = dict(np.load(cache, allow_pickle=True))
    for split_files in (train, val):
        for f in split_files:
            key = f'{f["sound"]}_{f["curve"]}'
            if key in lat: continue
            x = torch.from_numpy(f['audio'])[None, None].to(device)
            with torch.no_grad():
                lat[key] = rave.encode(x)[0].cpu().numpy()  # (D, F)
    np.savez(cache, **lat)
    D = next(iter(lat.values())).shape[0]
    print(f'latent dim D={D}, cached {len(lat)}')
    # per-dim latent normalization stats from train
    zall = np.concatenate([lat[f'{f["sound"]}_{f["curve"]}'] for f in train], axis=1)
    z_mean = zall.mean(axis=1, keepdims=True); z_std = zall.std(axis=1, keepdims=True) + 1e-6
    np.savez(os.path.join(OUT, 'zstats.npz'), mean=z_mean, std=z_std)
    def env_target(f, F):
        a = f['audio']
        fr = int(hop)
        n = F
        e = np.array([np.sqrt(np.mean(a[int(i*hop):int((i+1)*hop)]**2) + 1e-12) for i in range(n)])
        return np.log(e + 1e-4)
    env = {f'{f["sound"]}_{f["curve"]}': env_target(f, lat[f'{f["sound"]}_{f["curve"]}'].shape[1]) for f in train + val}

    # align sensor (100 Hz) to latent frames: frame i covers sample i*hop -> sensor idx i*hop/100
    def sensor_at_frames(f, F):
        idx = np.clip((np.arange(F) * hop / 100.0).astype(int), 0, len(f['sensor']) - 1)
        return f['sensor'][idx], f['touch'][idx]

    class Mapper(nn.Module):
        def __init__(self, n_sounds, d=16, hidden=256):
            super().__init__()
            self.emb = nn.Embedding(n_sounds, 32)
            self.net = nn.Sequential(
                nn.Linear(2 + 32, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
                nn.Linear(hidden, d))
            self.env_head = nn.Linear(hidden, 1)
        def forward(self, sensor, touch, sid):
            e = self.emb(sid)  # (B, 32)
            e = e.unsqueeze(1).expand(sensor.shape[0], sensor.shape[1], -1)  # (B,F,32)
            x = torch.cat([sensor, touch, e], dim=-1)
            h = self.net[:-1](x)
            return self.net[-1](h), self.env_head(h).squeeze(-1)  # z, log-rms env

    n_sounds = len({f['sound_id'] for f in train})
    mapper = Mapper(n_sounds, D).to(device)
    n_params = sum(p.numel() for p in mapper.parameters())
    print(f'mapper params: {n_params}')
    opt = torch.optim.Adam(mapper.parameters(), lr=5e-5)

    # training tensors: random windows of 256 latent frames
    WIN = 256
    steps = 2000
    for step in range(steps):
        batch = np.random.choice(len(train), size=8, replace=False)
        sensors, touches, sids, zs, s0s = [], [], [], [], []
        for bi in batch:
            f = train[bi]; key = f'{f["sound"]}_{f["curve"]}'
            z = lat[key]; F = z.shape[1]
            s0 = int(np.random.randint(0, max(F - WIN, 1)))
            s0 = min(s0, F - WIN)
            zs.append(torch.from_numpy(z[:, s0:s0+WIN]).to(device))
            s, t = sensor_at_frames(f, F)
            sensors.append(torch.from_numpy(s[s0:s0+WIN]))
            touches.append(torch.from_numpy(t[s0:s0+WIN]))
            sids.append(f['sound_id'])
            s0s.append(s0)
        sensor = torch.nan_to_num(torch.stack(sensors), 0.0, 0.0, 1.0).unsqueeze(-1).to(device)  # (B,F,1)
        touch = torch.nan_to_num(torch.stack(touches), 0.0, 0.0, 1e6).unsqueeze(-1).to(device)
        sid = torch.tensor(sids, device=device)
        zt = torch.stack(zs).permute(0, 2, 1).to(device)  # (B,F,D)
        zm = torch.nan_to_num(torch.from_numpy(z_mean).to(device).T, 0, 0, 0); zsd = torch.nan_to_num(torch.from_numpy(z_std).to(device).T, 1, 1, 1)
        zt_n = (zt - zm) / zsd
        et = []
        for b in range(len(batch)):
            k = f'{train[b]["sound"]}_{train[b]["curve"]}'
            seg = env[k][s0s[b]:s0s[b]+WIN]
            if len(seg) < WIN:
                seg = np.pad(seg, (0, WIN - len(seg)), mode='edge')
            et.append(torch.from_numpy(seg))
        et = torch.nan_to_num(torch.stack(et), -20.0, -20.0, 20.0).to(device)
        pred, penv = mapper(sensor, touch, sid)
        loss = (pred - zt_n).pow(2).mean()
        if not torch.isfinite(loss):
            keys = [train[b]['sound'] + '_' + train[b]['curve'] for b in range(len(batch))]
            print(f'step {step}: NaN loss, batch keys={keys}')
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0); opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f'step {step} latent_mse {float(loss):.4f}')
    torch.save(mapper.state_dict(), os.path.join(OUT, 'mapper_final.pth'))

    # eval on val
    from scipy.stats import pearsonr, spearmanr
    results = []
    for f in val:
        key = f'{f["sound"]}_{f["curve"]}'
        z = lat[key]; F = z.shape[1]
        s, t = sensor_at_frames(f, F)
        sensor = torch.from_numpy(s).unsqueeze(0).unsqueeze(-1).to(device)
        touch = torch.from_numpy(t).unsqueeze(0).unsqueeze(-1).to(device)
        sid = torch.tensor([f['sound_id']], device=device)
        mapper.eval()
        with torch.no_grad():
            zp = mapper(sensor, touch, sid)[0][0].cpu().numpy()  # (F,D)
            y = rave.decode(torch.from_numpy(zp).T.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        # ablation roll-2000 (sensor frames are 100 Hz -> 2000 frames = 20 s)
        with torch.no_grad():
            s_roll = np.roll(s, 2000)
            sensor_r = torch.from_numpy(s_roll).unsqueeze(0).unsqueeze(-1).to(device)
            zp_r = mapper(sensor_r, touch, sid)[0][0].cpu().numpy()
            y_r = rave.decode(torch.from_numpy(zp_r).T.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            s_zero = np.zeros_like(s)
            sensor_z = torch.from_numpy(s_zero).unsqueeze(0).unsqueeze(-1).to(device)
            zp_z = mapper(sensor_z, touch, sid)[0][0].cpu().numpy()
            y_z = rave.decode(torch.from_numpy(zp_z).T.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        tgt = f['audio']
        n = min(len(y), len(tgt)); yy, tt = y[:n], tgt[:n]
        with torch.no_grad():
            ml = float(compute_multiscale_loss(yy, tt, SCALES, OVERLAP, device))
            sl = float(compute_multiscale_loss(np.zeros_like(tt), tt, SCALES, OVERLAP, device))
        rr = float(np.sqrt((yy**2).mean() / ((tt**2).mean() + 1e-12)))
        abl_r = float(pearsonr(rms_env(yy), rms_env(y_r[:n]))[0])
        abl_z = float(pearsonr(rms_env(yy), rms_env(y_z[:n]))[0])
        env = rms_env(yy)
        env_ds = np.interp(np.linspace(0, len(env) - 1, F), np.arange(len(env)), env)
        spear_rms = float(spearmanr(s, env_ds)[0])
        results.append(dict(sound=f['sound'], curve=f['curve'], ms_loss=ml, silence=sl,
                            loss_ratio=ml/sl, rms_ratio=rr, peak=float(np.abs(yy).max()),
                            band_shape=band_shape_corr(yy, tt),
                            abl_roll2000=abl_r, abl_zero=abl_z, spearman_sensor_rms=spear_rms))
        print(f"{f['sound']}_{f['curve']}: loss_ratio={ml/sl:.3f} rms={rr:.3f} peak={np.abs(yy).max():.3f} band={band_shape_corr(yy,tt):.3f} abl_r={abl_r:.3f} abl_z={abl_z:.3f} spear={spear_rms:.3f}")
    json.dump(dict(mapper_params=n_params, latent_hop=float(hop), results=results),
              open(os.path.join(OUT, 'metrics.json'), 'w'), indent=2)
    agg = {k: float(np.mean([r[k] for r in results])) for k in
           ['loss_ratio', 'rms_ratio', 'peak', 'band_shape', 'abl_roll2000', 'abl_zero', 'spearman_sensor_rms']}
    print('MEAN:', {k: round(v, 4) for k, v in agg.items()})

if __name__ == '__main__':
    main()
