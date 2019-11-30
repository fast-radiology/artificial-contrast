from typing import List
import numpy as np
import torch

# monkey patch Path
from fastai.vision import *

from artificial_contrast.data import get_scans
from artificial_contrast.dicom import read_HU_array
from artificial_contrast.freqs import freqhist_bins, hist_scaled_img
from artificial_contrast.settings import SEED


DATA_PATH = Path(os.environ['DATA'])
FREQS = os.environ.get('FREQS', 'freqs.npy')
WINDOWS = [-100, 300]


def remove_outside_window(
    tensor: torch.Tensor, windows: List[int] = WINDOWS
) -> torch.Tensor:
    min_val, max_val = windows
    tensor = tensor[(tensor >= min_val) & (tensor <= max_val)]
    return tensor


scans = get_scans(DATA_PATH, test=False)
sample_dcms = torch.cat(
    tuple(
        [
            remove_outside_window(torch.tensor(read_HU_array(fn)), WINDOWS)
            for fn in sorted(scans)
        ]
    )
)

freqs = freqhist_bins(sample_dcms)
np.save(FREQS, freqs.numpy())

scaled_samples = torch.stack(
    tuple([hist_scaled_img(scan, freq_bins=freqs) for scan in scans])
)
# for normalization
mean, std = scaled_samples.mean(), scaled_samples.std()
print(mean, std)
