"""V9 tests: attack_lag_s kwarg / set_attack_lag, full onset_weighting pass-through."""
import torch
from engine_ddsp.sensor_ddsp import SensorDDSP

F0 = [110.0, 210.0, 100.0, 200.0, 110.0]
DEFAULT_LAG = [0.38, 0.38, 0.42, 0.245, 0.39]
NEW_LAG = [0.30, 0.31, 0.32, 0.20, 0.33]
OLD5 = {'enabled': True, 'factor': 5.0, 'window_ms': 150, 'dilate_frames': 1, 'touch_rate_hz': 100}


def _onset_touch(T=300, frame=100):
    touch = torch.zeros(1, T, 1)
    touch[:, frame, 0] = float(torch.log1p(torch.tensor(0.01)))  # fresh crossing at `frame`
    touch[:, frame + 1:, 0] = 1.0
    return touch


def test_default_buffer_unchanged():
    m = SensorDDSP(f0_base_hz=F0)
    assert torch.allclose(m.attack_lag_s, torch.tensor(DEFAULT_LAG)), m.attack_lag_s
    print('default attack_lag_s OK')


def test_kwarg_sets_buffer():
    m = SensorDDSP(f0_base_hz=F0, attack_lag_s=NEW_LAG)
    assert torch.allclose(m.attack_lag_s, torch.tensor(NEW_LAG)), m.attack_lag_s
    m2 = SensorDDSP(f0_base_hz=F0, attack_lag_s=torch.tensor(NEW_LAG, dtype=torch.float64))
    assert torch.allclose(m2.attack_lag_s, torch.tensor(NEW_LAG)), m2.attack_lag_s
    assert m2.attack_lag_s.dtype == torch.float32
    print('attack_lag_s kwarg OK')


def test_wrong_length_raises():
    for bad in ([0.3, 0.3], [0.3] * 6, torch.zeros(4)):
        try:
            SensorDDSP(f0_base_hz=F0, attack_lag_s=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'length {len(bad)} accepted')
        m = SensorDDSP(f0_base_hz=F0)
        try:
            m.set_attack_lag(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'set_attack_lag length {len(bad)} accepted')
    print('wrong length raises OK')


def test_set_attack_lag_device_dtype():
    m = SensorDDSP(f0_base_hz=F0)
    buf_id = id(m.attack_lag_s)
    m.set_attack_lag(torch.tensor(NEW_LAG, dtype=torch.float64))
    assert id(m.attack_lag_s) == buf_id, 'buffer replaced instead of copied in place'
    assert m.attack_lag_s.dtype == torch.float32
    assert torch.allclose(m.attack_lag_s, torch.tensor(NEW_LAG))
    m.set_attack_lag(DEFAULT_LAG)  # list input
    assert torch.allclose(m.attack_lag_s, torch.tensor(DEFAULT_LAG))
    assert m.set_attack_lag(NEW_LAG) is None
    if torch.cuda.is_available():
        m.to('cuda')
        m.set_attack_lag(torch.tensor(NEW_LAG, dtype=torch.float64))
        assert m.attack_lag_s.device.type == 'cuda' and m.attack_lag_s.dtype == torch.float32
        assert torch.allclose(m.attack_lag_s.cpu(), torch.tensor(NEW_LAG))
    print('set_attack_lag dtype/device OK')


def test_set_attack_lag_after_load_state_dict():
    m1 = SensorDDSP(f0_base_hz=F0)
    sd = m1.state_dict()
    sd['attack_lag_s'] = torch.tensor(DEFAULT_LAG)  # old checkpoint values
    m2 = SensorDDSP(f0_base_hz=F0, attack_lag_s=NEW_LAG)
    m2.load_state_dict(sd)
    assert torch.allclose(m2.attack_lag_s, torch.tensor(DEFAULT_LAG)), 'load should restore old values'
    m2.set_attack_lag(NEW_LAG)
    assert torch.allclose(m2.attack_lag_s, torch.tensor(NEW_LAG)), 'set_attack_lag did not restore new values'
    print('set_attack_lag after load_state_dict OK')


def test_onset_weighting_pass_through():
    m = SensorDDSP(f0_base_hz=F0, onset_weighting={'enabled': True, 'factor': 5.0, 'peak_ms': 80, 'pre_ms': 20, 'window_ms': 400})
    assert m.onset_weighting['peak_ms'] == 80, m.onset_weighting
    assert m.onset_weighting['pre_ms'] == 20, m.onset_weighting
    touch = _onset_touch(frame=100)
    w = m._onset_weight_profile_100hz(touch)[0, :, 0]
    pk = w.argmax().item()
    assert abs(pk - (100 + 8)) <= 1, f'profile peak at frame {pk}, want ~108 (80 ms)'
    print(f'onset peak_ms=80 pass-through OK (peak frame {pk})')


def test_old_five_key_call_unchanged():
    m = SensorDDSP(f0_base_hz=F0, onset_weighting=dict(OLD5))
    assert m.onset_weighting == OLD5, m.onset_weighting
    ref = SensorDDSP(f0_base_hz=F0, onset_weighting=dict(OLD5, peak_ms=350, pre_ms=100))
    touch = _onset_touch(frame=100)
    assert torch.allclose(m._onset_weight_profile_100hz(touch), ref._onset_weight_profile_100hz(touch))
    print('old 5-key onset_weighting call unchanged OK')


if __name__ == '__main__':
    test_default_buffer_unchanged()
    test_kwarg_sets_buffer()
    test_wrong_length_raises()
    test_set_attack_lag_device_dtype()
    test_set_attack_lag_after_load_state_dict()
    test_onset_weighting_pass_through()
    test_old_five_key_call_unchanged()
    print('ALL V9 ATTACK-LAG TESTS PASS')
