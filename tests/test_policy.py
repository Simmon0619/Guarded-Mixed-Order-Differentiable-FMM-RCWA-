from types import SimpleNamespace

import pytest
import torch

from workbench import descent, protocol


def make_policy(**guards):
    cfg = protocol.ProblemConfig(nx=9, ny=9)
    ctx = SimpleNamespace(cfg=cfg, mode="none", device="cpu", dtype=torch.complex128)
    return descent.GradPolicy(ctx, (1, 0), (2, 1), guards, k=3, mix_from_beta=None, mix_from_step=2)


def test_mixed_order_cadence_starts_at_requested_step():
    policy = make_policy()
    assert [t for t in range(11) if policy.is_high_step(t, 8)] == [2, 5, 8]


@pytest.mark.parametrize("stable,expected_action", [(True, "kept_original"), (False, "jitter")])
def test_norm_guard_distinguishes_stable_growth_from_changed_direction(monkeypatch, stable, expected_action):
    policy = make_policy()
    policy.norms[(1, 0)] = [1.0] * 4
    original = torch.tensor([10.0, 0.0], dtype=torch.float64)
    retry = original * 0.99 if stable else torch.tensor([0.0, 1.0], dtype=torch.float64)
    calls = iter([(0.5, original, 0.1, None), (0.51, retry, 0.1, None)])
    monkeypatch.setattr(policy, "_fwd_bwd", lambda *a, **kw: next(calls))
    _, gradient, row = policy.gradient(torch.zeros(2), (1, 0), 8, 4, False)
    assert row["action"] == expected_action
    torch.testing.assert_close(gradient, original if stable else retry)
    assert policy.stats["n_trigger"]["norm"] == 1


def test_disabled_norm_guard_logs_without_replacing(monkeypatch):
    policy = make_policy(enabled=False)
    policy.norms[(1, 0)] = [1.0] * 4
    original = torch.tensor([10.0, 0.0], dtype=torch.float64)
    monkeypatch.setattr(policy, "_fwd_bwd", lambda *a, **kw: (0.5, original, 0.1, None))
    _, gradient, row = policy.gradient(torch.zeros(2), (1, 0), 8, 4, False)
    assert row["action"] == "log_only_norm"
    assert policy.stats["n_would_trigger"]["norm"] == 1
    torch.testing.assert_close(gradient, original)


def test_failed_high_order_gradient_is_refused(monkeypatch):
    policy = make_policy()
    monkeypatch.setattr(policy, "_fwd_bwd", lambda *a, **kw: (None, None, float("nan"), "nonfinite"))
    value, gradient, row = policy.gradient(torch.zeros(2), (2, 1), 8, 4, True)
    assert value is None and gradient is None
    assert row["action"] == "skip_high"
    assert policy.c128_used == 0


def test_low_order_failure_uses_bounded_cpu_fallback(monkeypatch):
    policy = make_policy(c128_cap=1)
    gradient = torch.tensor([1.0, 2.0], dtype=torch.float64)
    def evaluate(*args, **kwargs):
        if kwargs.get("dtype") == torch.complex128:
            assert kwargs["device"] == "cpu"
            return 0.4, gradient, 0.1, None
        return None, None, float("nan"), "raise: injected failure"
    monkeypatch.setattr(policy, "_fwd_bwd", evaluate)
    _, actual, row = policy.gradient(torch.zeros(2), (1, 0), 8, 4, False)
    assert row["action"] == "c128"
    torch.testing.assert_close(actual, gradient)
    _, actual, row = policy.gradient(torch.zeros(2), (1, 0), 8, 5, False)
    assert actual is None and row["action"] == "skipped_update"
    assert policy.c128_used == 1


def test_checkpoint_selection_uses_gate_not_descent(monkeypatch):
    ctx = SimpleNamespace(cfg=protocol.ProblemConfig(nx=9, ny=9), mode="none", device="cpu")
    ctx.ops = SimpleNamespace(app_initial_latent=lambda *a: torch.zeros((9, 9)),
                              AppAdam=lambda *a: object(), app_hard_binary=lambda x, *a: x.clone())
    policy = SimpleNamespace(level_g=(2, 1), level_d=(1, 0), level_d0=(1, 0))
    # The second checkpoint has a larger low-order value but a lower gate value.
    readings = iter([0.8, 0.5, 0.6, 0.95])
    monkeypatch.setattr(descent, "eta_binary", lambda *a: next(readings))
    ticket = descent.TicketDescent(ctx, 1, 1, lambda *a: 2.0, 2.0, [0, 1], policy)
    ticket.latent += 1
    ticket._checkpoint(1)
    assert ticket.best["step"] == 0
    assert ticket.last["step"] == 1


def test_repair_chain_accepts_only_strict_gate_improvements(monkeypatch):
    cfg = protocol.ProblemConfig(nx=9, ny=9)
    ctx = SimpleNamespace(cfg=cfg, mode="none", device="cpu", dtype=torch.complex128)
    ctx.ops = SimpleNamespace(filtered_noise=lambda *a: torch.zeros((9, 9)),
                              sym=lambda x, *a: x, AppAdam=lambda *a: object(),
                              app_hard_binary=lambda x, *a: x.clone())
    # Reject a worse candidate and a tie, then accept a gate improvement.
    # A lower higher-order value is tracked as a false accept, not hidden.
    readings = iter([0.6, 0.8, 0.9, 0.65])
    monkeypatch.setattr(descent, "eta_binary", lambda *a: next(readings))
    parent = torch.zeros((9, 9))
    record, _ = descent.repair_chain(ctx, parent, parent, 0.8, 0.7, 64,
                                     (1, 0), (2, 1), (3, 1),
                                     {"hops": 3, "re_steps": 0, "stop_after": 3}, {}, 9)
    assert [r["accepted"] for r in record["jumps"]] == [False, False, True]
    assert record["n_accept"] == 1
    assert record["n_false_accept"] == 1
    assert record["final_eta_gate"] == 0.9
