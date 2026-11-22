"""V8b unit tests: onset window direction, all-scales coverage, RMS no-crash, ckpt load with fast_head."""
import torch
from engine_ddsp.sensor_ddsp import SensorDDSP, _onset_from_touch

F0 = [110.0, 210.0, 100.0, 200.0, 110.0]
SCALES = [4096, 2048, 1024, 512, 256, 128]


def test_window_direction():
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': True, 'factor': 5.0, 'window_ms': 800, 'peak_ms': 350, 'pre_ms': 100})
    touch = torch.zeros(1, 1000, 1)
    touch[:, 500, 0] = float(torch.log1p(torch.tensor(0.01)))  # fresh onset at frame 500
    touch[:, 501:, 0] = 1.0  # already in touch afterwards
    w = m._onset_weight_profile_100hz(touch)[0, :, 0]
    assert w[550] > 1.0 and w[555] > 1.0, 'no boost after onset'
    assert w[300] == 1.0, f'boost too early at frame 300: {w[300]}'
    assert w[900] < w[510], 'profile should decay after onset'
    print('window direction OK (causal, decaying)')


def test_all_scales_see_onset():
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': True, 'factor': 5.0, 'window_ms': 800, 'peak_ms': 350, 'pre_ms': 100})
    touch = torch.zeros(1, 6000, 1)
    touch[:, 3000, 0] = float(torch.log1p(torch.tensor(0.01)))
    touch[:, 3001:, 0] = 1.0
    w100 = m._onset_weight_profile_100hz(touch)
    for s in SCALES:
        hop = int(s * 0.25)
        F_ = (6000 * 480) // hop
        w_s = m._profile_to_stft(w100, F_, hop)
        assert (w_s > 1.01).any(), f'scale {s}: onset dropped (max={w_s.max().item():.3f})'
    print('all 6 STFT scales see the onset OK')


def test_rms_mask_no_crash():
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': True, 'factor': 5.0, 'window_ms': 800, 'peak_ms': 350, 'pre_ms': 100})
    touch = torch.zeros(1, 200, 3)
    touch[:, 50, 0] = float(torch.log1p(torch.tensor(0.01)))
    touch[:, 51:, 0] = 1.0
    loss, det = m.loss_curriculum(torch.randn(1, 48000 * 2), torch.randn(1, 48000 * 2),
                                  SCALES, 0.75, 1000, touch=touch, sound_id=torch.tensor([0]),
                                  return_losses=True)
    assert torch.isfinite(loss), 'NaN loss'
    print(f'loss_curriculum OK: loss={loss.item():.2f}, rms={det["rms"].item():.4f}')


def test_checkpoint_load_with_fast_head():
    m1 = SensorDDSP(f0_base_hz=F0)
    sd = {k: v for k, v in m1.state_dict().items() if not k.startswith('fast_head.')}  # simulate V6 ckpt
    m2 = SensorDDSP(f0_base_hz=F0)
    m2.load_state_dict(sd)  # must not raise: missing fast_head injected by hook
    assert (m2.fast_head.weight == 0).all(), 'fast_head should stay zero after V6 ckpt load'
    print('checkpoint load with fast_head OK (zero-init preserved)')


def test_onset_helper_no_frame0_false_positive():
    touch = torch.zeros(1, 100, 1)
    touch[:, 10:, 0] = 0.5  # mid-chunk touch start at frame 10... 
    # simulate chunk starting mid-touch: touch[0] > 0 but t_sec large
    touch2 = torch.zeros(1, 100, 1)
    touch2[:, :] = float(torch.log1p(torch.tensor(5.0)))  # already 5s in touch at frame 0
    on = _onset_from_touch(touch2)
    assert on[0, 0, 0] == 0, 'false onset at frame 0 of mid-touch chunk'
    touch3 = torch.zeros(1, 100, 1)
    touch3[:, 10, 0] = float(torch.log1p(torch.tensor(0.005)))  # fresh onset at frame 10
    touch3[:, 11:, 0] = 1.0
    on3 = _onset_from_touch(touch3)
    assert on3[0, 10, 0] == 1 and on3[0, 11, 0] == 0
    print('onset helper: no frame-0 false positive, detects fresh onset OK')


if __name__ == '__main__':
    test_window_direction()
    test_all_scales_see_onset()
    test_rms_mask_no_crash()
    test_checkpoint_load_with_fast_head()
    test_onset_helper_no_frame0_false_positive()
    print('ALL V8B TESTS PASS')
