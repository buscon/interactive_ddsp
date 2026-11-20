"""Gate: pretrained RAVE .ts recon of held-out Sentire val files, scored with Stage B metrics."""
import os, sys, json, glob
import numpy as np
import torch

REPO = '/home/marcello/Documents/interactive_ddsp'
sys.path.insert(0, REPO)
from eval_stageB import compute_multiscale_loss

SR = 48000
SCALES = [4096, 2048, 1024, 512, 256, 128]; OVERLAP = 0.75  # from config_sensor.yaml

def band_shape_corr(pred, tgt, bands=[(0,500),(500,2000),(2000,6000),(6000,12000),(12000,24000)]):
    def mean_log_band(x):
        x = np.asarray(x).ravel()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return np.log(np.array([np.sqrt(spec[(f>=lo)&(f<hi)].sum()+1e-12) for lo,hi in bands]) + 1e-6)
    from scipy.stats import pearsonr
    return float(pearsonr(mean_log_band(pred), mean_log_band(tgt))[0])

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = sorted(glob.glob('/home/marcello/Documents/RAVE_SC_claude/models/*.ts'))
    idx = json.load(open(os.path.join(REPO, 'bakeoff_data_aligned/index.json')))
    val = [(s, c) for s in sorted(idx['sound_ids'], key=lambda k: idx['sound_ids'][k]) for c in idx['val_curves']]
    files = {}
    for s, c in val:
        p = os.path.join(REPO, 'bakeoff_data_aligned', f'{s}_proximity_curve_{c}.npz')
        if os.path.exists(p):
            d = np.load(p)
            files[(s, c)] = d['audio'].astype(np.float32) / 32768.0
    print(f'{len(files)} val files, {len(models)} models, device={device}')
    results = {}
    for mp in models:
        name = os.path.basename(mp).replace('.ts', '')
        m = torch.jit.load(mp, map_location=device).eval()
        rows = []
        for (s, c), audio in sorted(files.items()):
            x = torch.from_numpy(audio)[None, None].to(device)
            with torch.no_grad():
                z = m.encode(x)
                y = m.decode(z).squeeze().cpu().numpy()
            n = min(len(y), len(audio)); y, t = y[:n], audio[:n]
            with torch.no_grad():
                ml = float(compute_multiscale_loss(y, t, SCALES, OVERLAP, device))
                sl = float(compute_multiscale_loss(np.zeros_like(t), t, SCALES, OVERLAP, device))
            rr = float(np.sqrt((y**2).mean() / ((t**2).mean() + 1e-12)))
            rows.append(dict(sound=s, curve=c, ms_loss=ml, silence=sl, loss_ratio=ml/sl,
                             rms_ratio=rr, peak=float(np.abs(y).max()), band_shape=band_shape_corr(y, t),
                             z_frames=int(z.shape[-1])))
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ['loss_ratio','rms_ratio','peak','band_shape']}
        results[name] = dict(per_file=rows, mean=agg)
        print(f'{name}: loss_ratio={agg["loss_ratio"]:.3f} rms={agg["rms_ratio"]:.3f} peak={agg["peak"]:.3f} band={agg["band_shape"]:.3f}')
    out = os.path.join(os.path.dirname(__file__), '..', 'runs', 'rave_gate', 'gate_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2)
    print('wrote', out)

if __name__ == '__main__':
    main()
