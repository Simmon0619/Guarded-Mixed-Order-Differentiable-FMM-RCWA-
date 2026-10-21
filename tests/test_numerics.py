import math

import pytest
import torch

from workbench import app
from workbench.selfcheck import slab_transmission
from trustworthy_fmm.fourier import convolution
from trustworthy_fmm.solvers import vector_2d
from trustworthy_fmm.solvers.fresnel import homogeneous_stack


@pytest.mark.parametrize("theta,polarization", [(0.0, "x"), (17.0, "x"), (17.0, "y")])
def test_uniform_slab_matches_independent_transfer_matrix(slab_config, theta, polarization):
    params = dict(slab_config["params"], polarization=polarization)
    cfg = app.build_cfg(params)
    density = torch.ones(cfg.shape, dtype=torch.float64)
    result = app.solve_density(density, (2, 1), cfg, theta, 0, torch.complex128, "cpu")
    reference = homogeneous_stack(cfg.n_incident, [math.sqrt(cfg.eps_feature)], [cfg.thickness_nm],
                                  cfg.n_output, cfg.wavelength_nm, theta, "p" if polarization == "x" else "s")
    assert float(result.total_T) == pytest.approx(float(reference.T), abs=1e-9)
    assert float(result.total_R) == pytest.approx(float(reference.R), abs=1e-9)
    assert float(result.energy_residual) < 1e-9


def test_patterned_lossless_energy_and_zero_thickness(optimization_config):
    cfg = app.build_cfg(optimization_config["params"])
    binary = app.app_hard_binary(app.app_initial_latent(2029, cfg, "y", "cpu"), cfg, "y")
    result = app.solve_density(binary, (2, 1), cfg, 0, 0, torch.complex128, "cpu")
    assert abs(float(result.total_T + result.total_R) - 1) < 1e-8
    zero = app.solve_density(binary, (2, 1), cfg.variant(thickness_nm=0), 0, 0, torch.complex128, "cpu")
    expected = 4 * cfg.n_incident * cfg.n_output / (cfg.n_incident + cfg.n_output) ** 2
    assert float(zero.total_T) == pytest.approx(expected, abs=1e-9)


def gradient_case():
    cfg = app.build_cfg({"nx": 21, "ny": 17, "lx_nm": 1030, "ly_nm": 470,
                         "thickness_nm": 230, "n_feature": 2.4, "filter_nm": 45})
    latent = app.app_initial_latent(43, cfg, "none", "cpu")
    return cfg, latent


def test_autograd_directional_derivative_against_central_difference():
    cfg, latent = gradient_case()
    level, target, beta = (2, 1), (1, 0), 3.0
    value, grad = app.value_grad(latent, level, target, beta, cfg, "none", 11, 19, torch.complex128, "cpu", False)
    direction = torch.randn(latent.shape, dtype=latent.dtype, generator=torch.Generator().manual_seed(103))
    direction /= direction.norm()
    analytic = float((grad * direction).sum())
    assert math.isfinite(value) and abs(analytic) > 1e-8
    def forward(x):
        with torch.no_grad():
            density = app.app_projected(x, beta, cfg, "none")
            return app.eta_of(app.solve_density(density, level, cfg, 11, 19, torch.complex128, "cpu"), target)
    for h in (1e-3, 3e-4, 1e-4):
        finite_difference = (forward(latent + h * direction) - forward(latent - h * direction)) / (2 * h)
        assert finite_difference == pytest.approx(analytic, rel=2e-4, abs=2e-8)


def test_memory_path_preserves_value_and_gradient(monkeypatch):
    cfg, latent = gradient_case()
    args = (latent, (2, 1), (1, 0), 3.0, cfg, "none", 11, 19, torch.complex128, "cpu", False)
    monkeypatch.setattr(vector_2d, "MEMORY_EFFICIENT", False)
    v_ref, g_ref = app.value_grad(*args)
    monkeypatch.setattr(vector_2d, "MEMORY_EFFICIENT", True)
    value, grad = app.value_grad(*args)
    assert value == pytest.approx(v_ref, abs=1e-10)
    torch.testing.assert_close(grad, g_ref, rtol=1e-7, atol=1e-9)


def test_vectorized_convolution_matches_loop_and_gradient(monkeypatch):
    samples = torch.rand((9, 13), dtype=torch.float64, generator=torch.Generator().manual_seed(7)).requires_grad_(True)
    monkeypatch.setattr(convolution, "VECTORIZED_CONV", False)
    reference = convolution.convolution_matrix_2d(samples, (2, 1))
    grad_ref, = torch.autograd.grad(reference.abs().square().sum(), samples)
    monkeypatch.setattr(convolution, "VECTORIZED_CONV", True)
    actual = convolution.convolution_matrix_2d(samples, (2, 1))
    grad, = torch.autograd.grad(actual.abs().square().sum(), samples)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(grad, grad_ref, rtol=1e-12, atol=1e-12)


def test_uniform_slab_order_ladder(slab_config):
    cfg = app.build_cfg(slab_config["params"])
    expected = slab_transmission(1, 2, 1, 200, 600)
    for level in ((1, 0), (2, 1), (3, 2)):
        r = app.solve_density(torch.ones(cfg.shape, dtype=torch.float64), level, cfg, 0, 0, torch.complex128, "cpu")
        assert app.eta_of(r, (0, 0)) == pytest.approx(expected, abs=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_value_and_gradient_agree_with_cpu():
    cfg, latent = gradient_case()
    common = ((2, 1), (1, 0), 3.0, cfg, "none", 11, 19)
    vc, gc = app.value_grad(latent, *common, torch.complex128, "cpu", False)
    vg, gg = app.value_grad(latent.cuda(), *common, torch.complex64, "cuda", False)
    assert vg == pytest.approx(vc, abs=1e-5, rel=1e-3)
    torch.testing.assert_close(gg.cpu(), gc, rtol=1e-2, atol=1e-5)
