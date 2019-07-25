import numpy as np

BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]
NORMALIZATION_MODE = ["crop", "pad"]


def bayer_unify(raw: np.ndarray, input_pattern: str, target_pattern: str, mode: str) -> np.ndarray:
    """
    Convert a bayer raw image from one bayer pattern to another.

    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be unified.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    target_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The expected output pattern.
    mode: {"crop", "pad"}
        The way to handle submosaic shift. "crop" abandon the outmost pixels,
        and "pad" introduce extra pixels. Use "crop" in training and "pad" in
        testing.
    """
    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if target_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown target bayer pattern!')
    if mode not in NORMALIZATION_MODE:
        raise ValueError('Unknown normalization mode!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray')

    if input_pattern == target_pattern:
        h_offset, w_offset = 0, 0
    elif input_pattern[0] == target_pattern[2] and input_pattern[1] == target_pattern[3]:
        h_offset, w_offset = 1, 0
    elif input_pattern[0] == target_pattern[1] and input_pattern[2] == target_pattern[3]:
        h_offset, w_offset = 0, 1
    elif input_pattern[0] == target_pattern[3] and input_pattern[1] == target_pattern[2]:
        h_offset, w_offset = 1, 1
    else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
        raise RuntimeError('Unexpected pair of input and target bayer pattern!')

    if mode == "pad":
        out = np.pad(raw, [[h_offset, h_offset], [w_offset, w_offset]], 'reflect')
    elif mode == "crop":
        h, w = raw.shape
        out = raw[h_offset:h - h_offset, w_offset:w - w_offset]
    else:
        raise ValueError('Unknown normalization mode!')

    return out


def bayer_aug(raw: np.ndarray, flip_h: bool, flip_w: bool, transpose: bool, input_pattern: str) -> np.ndarray:
    """
    Apply augmentation to a bayer raw image.

    Parameters
    ----------
    raw : np.ndarray in shape (H, W)
        Bayer raw image to be augmented. H and W must be even numbers.
    flip_h : bool
        If True, do vertical flip.
    flip_w : bool
        If True, do horizontal flip.
    transpose : bool
        If True, do transpose.
    input_pattern : {"RGGB", "BGGR", "GRBG", "GBRG"}
        The bayer pattern of the input image.
    """

    if input_pattern not in BAYER_PATTERNS:
        raise ValueError('Unknown input bayer pattern!')
    if not isinstance(raw, np.ndarray) or len(raw.shape) != 2:
        raise ValueError('raw should be a 2-dimensional numpy.ndarray')
    if raw.shape[0] % 2 == 1 or raw.shape[1] % 2 == 1:
        raise ValueError('raw should have even number of height and width!')

    aug_pattern, target_pattern = input_pattern, input_pattern

    out = raw
    if flip_h:
        out = out[::-1, :]
        aug_pattern = aug_pattern[2] + aug_pattern[3] + aug_pattern[0] + aug_pattern[1]
    if flip_w:
        out = out[:, ::-1]
        aug_pattern = aug_pattern[1] + aug_pattern[0] + aug_pattern[3] + aug_pattern[2]
    if transpose:
        out = out.T
        aug_pattern = aug_pattern[0] + aug_pattern[2] + aug_pattern[1] + aug_pattern[3]

    out = bayer_unify(out, aug_pattern, target_pattern, "crop")
    return out
