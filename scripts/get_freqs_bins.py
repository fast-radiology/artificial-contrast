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

scans = get_scans(DATA_PATH, test=False)
sample_dcms = torch.stack(
    tuple([torch.tensor(read_HU_array(fn)) for fn in sorted(scans)])
)

if WINDOWS:
    min_val, max_val = WINDOWS
    sample_dcms = sample_dcms[(sample_dcms >= min_val) & (sample_dcms <= max_val)]

freqs = freqhist_bins(sample_dcms)
np.save(FREQS, freqs.numpy())

scaled_samples = torch.stack(
    tuple([hist_scaled_img(scan, freq_bins=freqs) for scan in scans])
)
# for normalization
mean, std = scaled_samples.mean(), scaled_samples.std()
print(mean, std)
