"""Tests that ensemble metrics honour `ens_dim` rather than assuming dim 1."""

import math

import pytest
import torch

from metrics.metrics import crps_ens, spread_skill_ratio, spread_squared

M, D, X, Y = 7, 2, 8, 8


@pytest.fixture
def ens_and_target():
    torch.manual_seed(0)
    pred = torch.randn(M, D, X, Y)   # (M, D, X, Y), i.e. ens_dim=0
    target = torch.randn(D, X, Y)
    return pred, target


@pytest.mark.parametrize("metric", [spread_skill_ratio, crps_ens])
def test_ens_dim_0_and_1_agree(ens_and_target, metric):
    """The same ensemble, laid out on dim 0 or dim 1, must score identically."""
    pred, target = ens_and_target
    from_dim0 = metric(pred, target, ens_dim=0)
    # (M, D, X, Y) -> (1, M, D, X, Y): a batch of one, ensemble on dim 1
    from_dim1 = metric(pred.unsqueeze(0), target.unsqueeze(0), ens_dim=1)
    assert torch.allclose(from_dim0, from_dim1.squeeze(0), atol=1e-6)


def test_spread_skill_correction_uses_ensemble_size(ens_and_target):
    """The correction factor must be sqrt((M+1)/M), not sqrt((D+1)/D)."""
    pred, target = ens_and_target

    spread = torch.sqrt(spread_squared(pred, ens_dim=0))
    skill = torch.sqrt(
        ((pred.mean(dim=0) - target) ** 2).mean()
    )
    expected = math.sqrt((M + 1) / M) * (spread / skill)

    assert torch.allclose(spread_skill_ratio(pred, target, ens_dim=0),
                          expected, atol=1e-6)

    # The test needs M != D to distinguish the two.
    wrong = math.sqrt((D + 1) / D) * (spread / skill)
    assert not torch.allclose(expected, wrong, atol=1e-6), (
        "test is vacuous unless M != D"
    )
