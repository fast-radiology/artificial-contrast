from typing import Callable, List

import torch
from torch import Tensor

from fastai.vision import Image

from artificial_contrast.dicom import read_HU_array
from artificial_contrast.const import FREQS, WINDOWS


def freqhist_bins(examination_tensor: Tensor, n_bins: int = 100):
    """
    Use it to generate frequency histogram
    """
    sorted_values = examination_tensor.sort()[0]
    t = torch.cat(
        [
            Tensor([0.001]),
            torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
            Tensor([0.999]),
        ]
    )
    t = (len(sorted_values) * t).long()
    return sorted_values[t].unique()


def remove_voxels_outside_window(
    tensor: torch.Tensor, windows: List[int] = None
) -> torch.Tensor:
    if windows:
        min_val, max_val = windows
        tensor = tensor[(tensor >= min_val) & (tensor <= max_val)]
    return tensor.view(-1)


def hist_scaled_img(
    path: str, freq_bins: Tensor = None, min_px: float = None, max_px: float = None
):
    px = Tensor(read_HU_array(path))
    if min_px is not None:
        px[px < min_px] = min_px
    if max_px is not None:
        px[px > max_px] = max_px
    return hist_scaled(px, freq_bins=freq_bins)


def hist_scaled(tensor_img: Tensor, freq_bins: Tensor = None):
    ys = torch.linspace(0.0, 1.0, len(freq_bins))
    return (
        interp_1d(tensor_img.flatten(), freq_bins, ys)
        .reshape(tensor_img.shape)
        .clamp(0.0, 1.0)
    )


def interp_1d(x: Tensor, xp: Tensor, fp: Tensor):
    "Same as `np.interp`"
    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    incx = fp[:-1] - (slopes * xp[:-1])
    locs = (x[:, None] >= xp[None, :]).long().sum(1) - 1
    locs = locs.clamp(0, len(slopes) - 1)
    return slopes[locs] * x + incx[locs]


def get_freqs_array(path: str, conf: Dict[str, object]) -> torch.Tensor:
    min_val, max_val = conf[WINDOWS]
    freqs = torch.tensor(conf[FREQS])
    return hist_scaled_img(path, freqs, min_val, max_val)


def open_dcm_img_factory(conf: Dict[str, object]) -> Callable:
    def open_dcm_image(path: str, *args, **kwargs) -> Image:
        arr = get_freqs_array(path, conf)
        return Image(arr.repeat(3, 1, 1))

    return open_dcm_image
