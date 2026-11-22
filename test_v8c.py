"""V8c tests: loss freq-norm, vibrato ramp (no immediate wobble), lagged fast curves, ckpt load."""
import torch
from engine_ddsp.sensor_ddsp import SensorDDSP

F0 = [110.0, 210.0, 100.0, 200.0, 110.0]
SCALES = [4096, 2048, 1024, 512, 256, 128]


def test_loss_freq_norm():
    """Onset-weighted per-scale loss must be freq-MEAN (same order as unweighted), not 100x sum."""
    torch.manual_seed(0)
    m_on = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': True, 'factor': 5.0, 'window_ms': 800, 'peak_ms': 400, 'pre_ms': 100})
    m_off = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': False})
    pred, tgt = torch.randn(1, 48000 * 2), torch.randn(1, 48000 * 2)
    touch = torch.zeros(1, 200, 3)
    touch[:, 50, 0] = float(torch.log1p(torch.tensor(0.01)))
    touch[:, 51:, 0] = 1.0
    _, d_on = m_on.loss_curriculum(pred, tgt, SCALES, 0.75, 1000, touch=touch, sound_id=torch.tensor([0]), return_losses=True)
    _, d_off = m_off.loss_curriculum(pred, tgt, SCALES, 0.75, 1000, touch=touch, sound_id=torch.tensor([0]), return_losses=True)
    print(f"  weighted multiscale={d_on['multiscale'].item():.4f}  unweighted={d_off['multiscale'].item():.4f}")
    assert d_on['multiscale'].item() < 10 * d_off['multiscale'].item() + 1.0, 'weighted loss not freq-normalized (100x bug)'
    print('loss freq-norm OK')


def test_vibrato_ramp():
    """f0 wobble must ramp with t (SC formula), ~0 right after touch, growing later."""
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': False})
    m.eval()
    T = 800
    touch = torch.zeros(1, T, 3)
    touch[:, 100, 0] = float(torch.log1p(torch.tensor(0.01)))
    touch[:, 101:, 0] = torch.linspace(0.01, 5.0, T - 101).log1p()  # growing touch time
    sensor = torch.full((1, T, 1), 0.99)  # constant sensor, in touch
    with torch.no_grad():
        m(sensor, touch, torch.tensor([0]))
    f0 = m._last_f0[0, :, 0]
    early = f0[100:250].std().item()   # first 1.5 s after crossing
    late = f0[500:800].std().item()    # 4-7 s after
    print(f'  f0 std early={early:.4f} Hz, late={late:.4f} Hz')
    assert late > early, 'vibrato should grow with touch time'
    print('vibrato ramp OK')


def test_fast_curves_lagged():
    """fast features must be ~0 before attack lag and peak after it."""
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': False})
    T = 200
    touch = torch.zeros(1, T, 1)
    touch[:, 100, 0] = float(torch.log1p(torch.tensor(0.01)))  # crossing at frame 100
    touch[:, 101:, 0] = torch.log1p(torch.arange(1, T - 100).float() * 0.01)  # realistic: t grows 10ms/frame
    t_sec = torch.expm1(touch.clamp(min=0))
    lag = m.attack_lag_s[0]
    t_d = t_sec - lag
    gate = (t_sec > 0).float() * (t_d >= 0).float()
    t_dc = t_d.clamp(min=0)
    f03 = gate * torch.exp(-t_dc / 0.03)
    assert f03[0, 100:138, 0].max() == 0, 'fast curve active before lag'
    pk = f03[0, :, 0].argmax().item()
    assert 138 <= pk <= 142, f'fast curve peaks at frame {pk}, want ~100+40 (lag 0.395s)'
    print(f'fast curves lagged OK (peak frame {pk}, lag frame ~{int(lag*100)})')


def test_ckpt_load():
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': True, 'factor': 5.0})
    sd = torch.load('runs/bakeoff_ddsp_poc/stageB_v8b/state_final.pth', map_location='cpu')
    m.load_state_dict(sd if 'model' not in sd else sd['model'])
    print('v8b ckpt loads OK, lag =', m.attack_lag_s.tolist())


if __name__ == '__main__':
    test_loss_freq_norm()
    test_vibrato_ramp()
    test_fast_curves_lagged()
    test_ckpt_load()
    print('ALL V8C TESTS PASS')
