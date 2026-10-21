"""Differentiable design filters and held-out fabrication perturbations.

The periodic helpers in this module use the native ``(Ny, Nx)`` design layout.
They are intentionally separate from the non-periodic convenience functions:
unit-cell optimization must not acquire artificial boundaries from reflected or
zero padding.
"""

from typing import List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as functional
from scipy import ndimage


def gaussian_kernel1d(
    sigma_px: float,
    *,
    truncate: float = 3.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    radius = max(1, int(truncate * sigma_px + 0.5))
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (coordinates / sigma_px) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth(density: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Differentiable Gaussian smoothing for 1D or 2D density arrays."""
    kernel = gaussian_kernel1d(
        sigma_px, dtype=density.dtype, device=str(density.device)
    )
    radius = kernel.numel() // 2
    if density.ndim == 1:
        value = density[None, None, :]
        value = functional.pad(value, (radius, radius), mode="reflect")
        return functional.conv1d(value, kernel[None, None, :])[0, 0]
    if density.ndim == 2:
        value = density[None, None, :, :]
        value = functional.pad(value, (radius, radius, radius, radius), mode="reflect")
        horizontal = functional.conv2d(value, kernel[None, None, None, :])
        vertical = functional.conv2d(horizontal, kernel[None, None, :, None])
        return vertical[0, 0]
    raise ValueError("density must be one- or two-dimensional")


def periodic_gaussian_filter(
    density_yx: torch.Tensor,
    lattice_nm: Sequence[float],
    radius_nm: float,
) -> torch.Tensor:
    """Differentiable circular Gaussian filtering on a periodic ``(Ny,Nx)`` cell."""
    if density_yx.ndim != 2:
        raise ValueError("density must have shape (Ny,Nx)")
    if len(lattice_nm) != 2 or any(float(value) <= 0 for value in lattice_nm):
        raise ValueError("lattice_nm must contain two positive periods")
    if radius_nm <= 0:
        raise ValueError("radius_nm must be positive")
    ny, nx = density_yx.shape
    dx = float(lattice_nm[0]) / nx
    dy = float(lattice_nm[1]) / ny
    y = (
        torch.arange(ny, dtype=density_yx.dtype, device=density_yx.device)
        - (ny - 1) / 2
    ) * dy
    x = (
        torch.arange(nx, dtype=density_yx.dtype, device=density_yx.device)
        - (nx - 1) / 2
    ) * dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / float(radius_nm) ** 2)
    kernel = kernel / kernel.sum()
    kernel_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(kernel)))
    density_fft = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(density_yx))
    )
    return torch.real(
        torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(density_fft * kernel_fft))
        )
    )


def tanh_projection(density: torch.Tensor, beta: float, eta: float = 0.5) -> torch.Tensor:
    if beta <= 0 or not 0 < eta < 1:
        raise ValueError("beta must be positive and eta must be in (0, 1)")
    eta_tensor = torch.as_tensor(eta, dtype=density.dtype, device=density.device)
    denominator = torch.tanh(beta * eta_tensor) + torch.tanh(beta * (1 - eta_tensor))
    return (
        torch.tanh(beta * eta_tensor) + torch.tanh(beta * (density - eta_tensor))
    ) / denominator


def spatial_tanh_projection(
    density: torch.Tensor,
    beta: float,
    eta: torch.Tensor,
) -> torch.Tensor:
    """Tanh projection with a spatial threshold field for soft CD/LER models."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    eta_tensor = torch.as_tensor(eta, dtype=density.dtype, device=density.device)
    if eta_tensor.ndim not in (0, density.ndim):
        raise ValueError("eta must be scalar or have the same rank as density")
    if eta_tensor.ndim > 0 and eta_tensor.shape != density.shape:
        raise ValueError("spatial eta must have the same shape as density")
    if bool(torch.any(eta_tensor <= 0)) or bool(torch.any(eta_tensor >= 1)):
        raise ValueError("every eta value must be in (0, 1)")
    denominator = torch.tanh(beta * eta_tensor) + torch.tanh(
        beta * (1 - eta_tensor)
    )
    return (
        torch.tanh(beta * eta_tensor)
        + torch.tanh(beta * (density - eta_tensor))
    ) / denominator


def correlated_periodic_field(
    shape_yx: Sequence[int],
    lattice_nm: Sequence[float],
    correlation_nm: float,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a deterministic zero-mean, unit-RMS periodic Gaussian field."""
    if len(shape_yx) != 2 or any(int(value) < 2 for value in shape_yx):
        raise ValueError("shape_yx must contain two sizes >= 2")
    if len(lattice_nm) != 2 or any(float(value) <= 0 for value in lattice_nm):
        raise ValueError("lattice_nm must contain two positive periods")
    if correlation_nm <= 0:
        raise ValueError("correlation_nm must be positive")
    ny, nx = (int(value) for value in shape_yx)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    white = torch.randn((ny, nx), generator=generator, dtype=torch.float64)
    fy = torch.fft.fftfreq(ny, d=float(lattice_nm[1]) / ny, dtype=torch.float64)
    fx = torch.fft.fftfreq(nx, d=float(lattice_nm[0]) / nx, dtype=torch.float64)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    spectrum_filter = torch.exp(
        -2.0
        * np.pi**2
        * float(correlation_nm) ** 2
        * (fxx.square() + fyy.square())
    )
    field = torch.real(torch.fft.ifft2(torch.fft.fft2(white) * spectrum_filter))
    field = field - field.mean()
    rms = torch.sqrt(torch.mean(field.square()))
    if float(rms) <= 1e-15:
        raise RuntimeError("correlated field lost all non-constant modes")
    return (field / rms).to(dtype=dtype, device=device)


def soft_dilate(density: torch.Tensor, radius_px: int) -> torch.Tensor:
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if radius_px == 0:
        return density
    size = 2 * radius_px + 1
    if density.ndim == 1:
        return functional.max_pool1d(
            density[None, None, :], size, stride=1, padding=radius_px
        )[0, 0]
    if density.ndim == 2:
        return functional.max_pool2d(
            density[None, None, :, :], size, stride=1, padding=radius_px
        )[0, 0]
    raise ValueError("density must be one- or two-dimensional")


def soft_erode(density: torch.Tensor, radius_px: int) -> torch.Tensor:
    return -soft_dilate(-density, radius_px)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    return ndimage.distance_transform_edt(binary) - ndimage.distance_transform_edt(~binary)


def periodic_signed_distance_nm(
    mask: np.ndarray,
    pixel_size_yx_nm: Sequence[float],
) -> np.ndarray:
    """Periodic signed distance in nm, positive inside the binary feature."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("periodic signed distance requires a 2D mask")
    if len(pixel_size_yx_nm) != 2 or any(
        float(value) <= 0 for value in pixel_size_yx_nm
    ):
        raise ValueError("pixel_size_yx_nm must contain two positive values")
    scale = max(binary.shape[0] * float(pixel_size_yx_nm[0]), binary.shape[1] * float(pixel_size_yx_nm[1]))
    if bool(binary.all()):
        return np.full(binary.shape, scale, dtype=np.float64)
    if not bool(binary.any()):
        return np.full(binary.shape, -scale, dtype=np.float64)
    tiled = np.tile(binary, (3, 3))
    inside = ndimage.distance_transform_edt(
        tiled, sampling=tuple(float(value) for value in pixel_size_yx_nm)
    )
    outside = ndimage.distance_transform_edt(
        ~tiled, sampling=tuple(float(value) for value in pixel_size_yx_nm)
    )
    signed = inside - outside
    ny, nx = binary.shape
    return signed[ny : 2 * ny, nx : 2 * nx]


def periodic_binary_morphology(
    mask: np.ndarray,
    operation: str,
    radius_nm: float,
    pixel_size_yx_nm: Sequence[float],
) -> np.ndarray:
    """Apply a physical-radius opening or closing with periodic boundaries."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("periodic morphology requires a 2D mask")
    if operation not in ("none", "opening", "closing"):
        raise ValueError("operation must be none, opening, or closing")
    if operation == "none" or radius_nm == 0:
        return binary.copy()
    if radius_nm < 0:
        raise ValueError("radius_nm must be non-negative")
    dy, dx = (float(value) for value in pixel_size_yx_nm)
    ry = max(1, int(np.ceil(radius_nm / dy)))
    rx = max(1, int(np.ceil(radius_nm / dx)))
    yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    structure = (yy * dy) ** 2 + (xx * dx) ** 2 <= radius_nm**2
    tiled = np.tile(binary, (3, 3))
    if operation == "opening":
        changed = ndimage.binary_opening(tiled, structure=structure)
    else:
        changed = ndimage.binary_closing(tiled, structure=structure)
    ny, nx = binary.shape
    return changed[ny : 2 * ny, nx : 2 * nx]


def hard_process_layers(
    nominal_mask: np.ndarray,
    scenario: Mapping[str, object],
    lattice_nm: Sequence[float],
) -> List[np.ndarray]:
    """Apply a held-out hard process scenario and return top-to-bottom masks.

    The catalog is exact only over its listed scenarios.  LER entries remain
    deterministic samples and are never labelled continuous worst cases.
    """
    binary = np.asarray(nominal_mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("nominal_mask must have shape (Ny,Nx)")
    ny, nx = binary.shape
    pixel_yx = (float(lattice_nm[1]) / ny, float(lattice_nm[0]) / nx)
    signed_nm = periodic_signed_distance_nm(binary, pixel_yx)
    cd_bias_nm = float(scenario.get("cd_bias_nm", 0.0))
    ler_rms_nm = float(scenario.get("ler_rms_nm", 0.0))
    if ler_rms_nm < 0:
        raise ValueError("ler_rms_nm must be non-negative")
    if ler_rms_nm > 0:
        correlation_nm = float(scenario.get("ler_correlation_nm", 0.0))
        field = correlated_periodic_field(
            binary.shape,
            lattice_nm,
            correlation_nm,
            int(scenario.get("ler_seed", 0)),
            dtype=torch.float64,
            device="cpu",
        ).numpy()
    else:
        field = np.zeros(binary.shape, dtype=np.float64)
    layers = int(scenario.get("sidewall_layers", 1))
    if layers < 1:
        raise ValueError("sidewall_layers must be positive")
    top = float(scenario.get("sidewall_top_bias_nm", 0.0))
    bottom = float(scenario.get("sidewall_bottom_bias_nm", 0.0))
    sidewall_biases = np.linspace(top, bottom, layers)
    outputs: List[np.ndarray] = []
    for sidewall_bias in sidewall_biases:
        local_offset = cd_bias_nm + float(sidewall_bias) + ler_rms_nm * field
        changed = signed_nm >= -local_offset
        changed = periodic_binary_morphology(
            changed,
            str(scenario.get("morphology", "none")),
            float(scenario.get("morphology_radius_nm", 0.0)),
            pixel_yx,
        )
        outputs.append(changed)
    return outputs


def sdf_smooth(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    return ndimage.gaussian_filter(signed_distance(mask), sigma=sigma_px) >= 0.0


def cd_bias(mask: np.ndarray, bias_nm: float, pixel_size_nm: float) -> np.ndarray:
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be positive")
    threshold_pixels = -bias_nm / pixel_size_nm
    return signed_distance(mask) >= threshold_pixels


def binary_opening_closing(mask: np.ndarray, radius_px: int) -> dict:
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if radius_px == 0:
        binary = np.asarray(mask, dtype=bool)
        return {"opening": binary.copy(), "closing": binary.copy()}
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim == 1:
        structure = np.ones(2 * radius_px + 1, dtype=bool)
    elif binary.ndim == 2:
        y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
        structure = x * x + y * y <= radius_px * radius_px
    else:
        raise ValueError("binary morphology supports one- or two-dimensional masks")
    return {
        "opening": ndimage.binary_opening(binary, structure=structure),
        "closing": ndimage.binary_closing(binary, structure=structure),
    }


def sidewall_stack(
    mask: np.ndarray,
    top_bias_nm: float,
    bottom_bias_nm: float,
    pixel_size_nm: float,
    layers: int = 3,
) -> List[np.ndarray]:
    if layers < 2:
        raise ValueError("sidewall stack requires at least two layers")
    biases = np.linspace(bottom_bias_nm, top_bias_nm, layers)
    return [cd_bias(mask, float(bias), pixel_size_nm) for bias in biases]


def held_out_perturbation_set(
    mask: np.ndarray,
    *,
    pixel_size_nm: float,
    morphology_radius_px: int = 3,
) -> dict:
    variants = {"nominal": np.asarray(mask, dtype=bool)}
    for sigma in (1.0, 2.0):
        variants["sdf_sigma_{}".format(int(sigma))] = sdf_smooth(mask, sigma)
    for bias in (-20.0, -10.0, 10.0, 20.0):
        variants["cd_bias_{:+.0f}nm".format(bias)] = cd_bias(mask, bias, pixel_size_nm)
    variants.update(binary_opening_closing(mask, morphology_radius_px))
    stack = sidewall_stack(mask, -20.0, 0.0, pixel_size_nm, layers=3)
    for index, layer in enumerate(stack):
        variants["sidewall_layer_{}".format(index)] = layer
    return variants
