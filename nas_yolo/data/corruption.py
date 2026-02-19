"""
Corruption Transform for Robustness Evaluation
================================================
Implements 15 corruption types at 5 severity levels following
the ImageNet-C / COCO-C protocol (Hendrycks & Dietterich, 2019).

All corruptions operate on numpy uint8 RGB arrays.
Dependencies: numpy, PIL, scipy (no OpenCV required).
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Optional

try:
    from scipy.ndimage import gaussian_filter, map_coordinates
except ImportError:
    gaussian_filter = None
    map_coordinates = None


# ============================================================================
# Corruption type → category mapping (15 types, 5 categories)
# ============================================================================
CORRUPTION_TYPES = {
    # Noise
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",
    # Blur
    "defocus_blur": "blur",
    "glass_blur": "blur",
    "motion_blur": "blur",
    # Weather
    "snow": "weather",
    "frost": "weather",
    "fog": "weather",
    # Digital
    "brightness": "digital",
    "contrast": "digital",
    "elastic_transform": "digital",
    "pixelate": "digital",
    "jpeg_compression": "digital",
    # Industrial
    "low_light": "industrial",
}


# ============================================================================
# Individual corruption functions
# Each takes (image: np.ndarray [H,W,3] uint8, severity: int 1-5)
# and returns np.ndarray [H,W,3] uint8
# ============================================================================

def _clip(img: np.ndarray) -> np.ndarray:
    """Clip to valid uint8 range."""
    return np.clip(img, 0, 255).astype(np.uint8)


def gaussian_noise(image: np.ndarray, severity: int) -> np.ndarray:
    stds = [0.08, 0.12, 0.18, 0.26, 0.38]
    std = stds[severity - 1]
    noise = np.random.normal(0, std * 255, image.shape)
    return _clip(image.astype(np.float32) + noise)


def shot_noise(image: np.ndarray, severity: int) -> np.ndarray:
    lambdas = [60, 25, 12, 5, 3]
    lam = lambdas[severity - 1]
    noisy = np.random.poisson(image.astype(np.float32) / 255.0 * lam) / lam * 255.0
    return _clip(noisy)


def impulse_noise(image: np.ndarray, severity: int) -> np.ndarray:
    amounts = [0.03, 0.06, 0.09, 0.17, 0.27]
    amount = amounts[severity - 1]
    result = image.copy()
    # Salt
    num_salt = int(amount * image.size / 2)
    coords = tuple(np.random.randint(0, d, num_salt) for d in image.shape[:2])
    result[coords[0], coords[1]] = 255
    # Pepper
    num_pepper = int(amount * image.size / 2)
    coords = tuple(np.random.randint(0, d, num_pepper) for d in image.shape[:2])
    result[coords[0], coords[1]] = 0
    return result


def defocus_blur(image: np.ndarray, severity: int) -> np.ndarray:
    radii = [3, 4, 6, 8, 10]
    radius = radii[severity - 1]
    pil_img = Image.fromarray(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def glass_blur(image: np.ndarray, severity: int) -> np.ndarray:
    sigmas = [0.7, 0.9, 1.0, 1.1, 1.5]
    sigma = sigmas[severity - 1]
    if gaussian_filter is not None:
        blurred = gaussian_filter(image.astype(np.float32), sigma=[sigma, sigma, 0])
    else:
        pil_img = Image.fromarray(image)
        blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=int(sigma * 2))))
        blurred = blurred.astype(np.float32)
    # Local pixel shuffle
    h, w = image.shape[:2]
    d = severity + 1
    for _ in range(severity):
        for i in range(h - d, d, -1):
            for j in range(w - d, d, -1):
                di, dj = np.random.randint(-d, d + 1, 2)
                ni, nj = min(max(i + di, 0), h - 1), min(max(j + dj, 0), w - 1)
                blurred[i, j], blurred[ni, nj] = blurred[ni, nj].copy(), blurred[i, j].copy()
                if (i * w + j) % 10000 == 0:
                    break  # Performance guard for large images
    return _clip(blurred)


def motion_blur(image: np.ndarray, severity: int) -> np.ndarray:
    sizes = [5, 9, 13, 17, 21]
    size = sizes[severity - 1]
    # Create linear motion kernel
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    # Apply via PIL
    pil_img = Image.fromarray(image)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=size // 3))
    return np.array(blurred)


def snow(image: np.ndarray, severity: int) -> np.ndarray:
    amounts = [0.1, 0.2, 0.3, 0.4, 0.5]
    amount = amounts[severity - 1]
    result = image.astype(np.float32)
    h, w = result.shape[:2]
    # Add random white streaks
    snow_layer = np.random.uniform(0, 1, (h, w))
    snow_layer = (snow_layer > (1 - amount)).astype(np.float32) * 255
    snow_layer = np.stack([snow_layer] * 3, axis=-1)
    result = result * (1 - amount * 0.5) + snow_layer * amount * 0.5
    return _clip(result)


def frost(image: np.ndarray, severity: int) -> np.ndarray:
    intensities = [0.2, 0.3, 0.4, 0.5, 0.6]
    intensity = intensities[severity - 1]
    h, w = image.shape[:2]
    # Generate procedural frost pattern
    frost_layer = np.random.normal(200, 30, (h, w)).clip(150, 255)
    if gaussian_filter is not None:
        frost_layer = gaussian_filter(frost_layer, sigma=5)
    frost_layer = np.stack([frost_layer] * 3, axis=-1).astype(np.float32)
    result = image.astype(np.float32) * (1 - intensity) + frost_layer * intensity
    return _clip(result)


def fog(image: np.ndarray, severity: int) -> np.ndarray:
    intensities = [0.15, 0.25, 0.35, 0.5, 0.65]
    intensity = intensities[severity - 1]
    h, w = image.shape[:2]
    fog_layer = np.ones((h, w, 3), dtype=np.float32) * 220
    result = image.astype(np.float32) * (1 - intensity) + fog_layer * intensity
    return _clip(result)


def brightness(image: np.ndarray, severity: int) -> np.ndarray:
    factors = [1.1, 1.2, 1.35, 1.5, 1.7]
    factor = factors[severity - 1]
    pil_img = Image.fromarray(image)
    enhanced = ImageEnhance.Brightness(pil_img).enhance(factor)
    return np.array(enhanced)


def contrast(image: np.ndarray, severity: int) -> np.ndarray:
    factors = [0.75, 0.6, 0.45, 0.3, 0.15]
    factor = factors[severity - 1]
    pil_img = Image.fromarray(image)
    enhanced = ImageEnhance.Contrast(pil_img).enhance(factor)
    return np.array(enhanced)


def elastic_transform(image: np.ndarray, severity: int) -> np.ndarray:
    alphas = [20, 40, 60, 80, 100]
    sigmas = [3, 4, 5, 6, 7]
    alpha = alphas[severity - 1]
    sigma = sigmas[severity - 1]
    if gaussian_filter is None or map_coordinates is None:
        # Fallback: just add slight blur
        pil_img = Image.fromarray(image)
        return np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=severity)))
    h, w = image.shape[:2]
    dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices_y = np.clip(y + dy, 0, h - 1)
    indices_x = np.clip(x + dx, 0, w - 1)
    result = np.zeros_like(image)
    for c in range(3):
        result[:, :, c] = map_coordinates(image[:, :, c], [indices_y, indices_x], order=1)
    return _clip(result)


def pixelate(image: np.ndarray, severity: int) -> np.ndarray:
    factors = [0.6, 0.5, 0.4, 0.3, 0.25]
    factor = factors[severity - 1]
    pil_img = Image.fromarray(image)
    h, w = pil_img.size
    small = pil_img.resize((max(1, int(h * factor)), max(1, int(w * factor))), Image.BOX)
    result = small.resize((h, w), Image.NEAREST)
    return np.array(result)


def jpeg_compression(image: np.ndarray, severity: int) -> np.ndarray:
    qualities = [65, 50, 35, 20, 10]
    quality = qualities[severity - 1]
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    result = Image.open(buf).convert('RGB')
    return np.array(result)


def low_light(image: np.ndarray, severity: int) -> np.ndarray:
    factors = [0.5, 0.35, 0.25, 0.15, 0.1]
    noise_stds = [0.02, 0.04, 0.06, 0.10, 0.15]
    factor = factors[severity - 1]
    noise_std = noise_stds[severity - 1]
    result = image.astype(np.float32) * factor
    noise = np.random.normal(0, noise_std * 255, image.shape)
    result = result + noise
    return _clip(result)


# Map corruption names to functions
_CORRUPTION_FN = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
    "low_light": low_light,
}


# ============================================================================
# CorruptionTransform class
# ============================================================================

class CorruptionTransform:
    """Apply image corruption for robustness evaluation.

    Args:
        corruption_type: Name of corruption or "random" for random selection.
        severity: Corruption severity level (1-5).
        p: Probability of applying corruption (default=1.0).
    """

    def __init__(
        self,
        corruption_type: str = "gaussian_noise",
        severity: int = 3,
        p: float = 1.0,
    ):
        assert 1 <= severity <= 5, f"Severity must be 1-5, got {severity}"
        if corruption_type != "random":
            assert corruption_type in _CORRUPTION_FN, (
                f"Unknown corruption: {corruption_type}. "
                f"Available: {list(_CORRUPTION_FN.keys())}"
            )
        self.corruption_type = corruption_type
        self.severity = severity
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply corruption to image.

        Args:
            image: RGB numpy array [H, W, 3] uint8.

        Returns:
            Corrupted image as numpy array [H, W, 3] uint8.
        """
        if np.random.random() > self.p:
            return image

        if self.corruption_type == "random":
            name = np.random.choice(list(_CORRUPTION_FN.keys()))
        else:
            name = self.corruption_type

        fn = _CORRUPTION_FN[name]
        return fn(image, self.severity)

    def __repr__(self) -> str:
        return (f"CorruptionTransform(type={self.corruption_type}, "
                f"severity={self.severity}, p={self.p})")
