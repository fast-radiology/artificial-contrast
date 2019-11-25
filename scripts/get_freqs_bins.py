import numpy as np

# monkey patch Path
from fastai.vision import *

from artifical_contrast.data import get_scans
from artifical_contrast.dicom import read_HU_array
from artifical_contrast.freqs import freqhist_bins, hist_scaled_img
from artificial_contrast.settings import SEED


DATA_PATH = os.environ['DATA']
FREQS_NAME = os.environ.get('FREQS_NAME', 'freqs.npy')
WINDOWS = [-100, 300]

scans = get_scans(DATA_PATH, test=False)
sample_dcms = torch.stack(tuple([read_HU_array(fn) for fn in sorted(scans)]))

if WINDOWS:
    min_val, max_val = WINDOWS
    sample_dcms[sample_dcms < min_val] = min_val
    sample_dcms[sample_dcms > max_val] = max_val

freqs = freqhist_bins(sample_dcms)
np.save(FREQS_NAME, freqs.numpy())

scaled_samples = torch.stack(
    tuple([hist_scaled_img(scan, freq_bins=freqs) for scan in scans])
)
# for normalization
mean, std = scaled_samples.mean(), scaled_samples.std()
print(mean, std)
