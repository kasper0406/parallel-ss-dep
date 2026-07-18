import pytest
import torch

from experiments.soup_ckpts import soup_state_dicts


def _sd(scale, dtype=torch.float32):
    return {
        "w": torch.full((3, 2), float(scale), dtype=dtype),
        "b": torch.full((2,), float(scale) * 2, dtype=dtype),
        "counter": torch.tensor(int(scale), dtype=torch.int64),
    }


def test_mean_of_three():
    out = soup_state_dicts([_sd(1.0), _sd(2.0), _sd(3.0)])
    assert torch.allclose(out["w"], torch.full((3, 2), 2.0))
    assert torch.allclose(out["b"], torch.full((2,), 4.0))


def test_non_float_from_first():
    out = soup_state_dicts([_sd(1.0), _sd(2.0)])
    assert out["counter"].item() == 1


def test_bf16_roundtrip_dtype_preserved():
    sds = [_sd(1.0, torch.bfloat16), _sd(2.0, torch.bfloat16)]
    out = soup_state_dicts(sds)
    assert out["w"].dtype == torch.bfloat16
    assert torch.allclose(out["w"].float(), torch.full((3, 2), 1.5), atol=1e-2)


def test_key_mismatch_raises():
    a, b = _sd(1.0), _sd(2.0)
    del b["b"]
    with pytest.raises(ValueError, match="key mismatch"):
        soup_state_dicts([a, b])


def test_shape_mismatch_raises():
    a, b = _sd(1.0), _sd(2.0)
    b["w"] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="shape mismatch"):
        soup_state_dicts([a, b])
