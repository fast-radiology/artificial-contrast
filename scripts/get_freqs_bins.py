import numpy as np

# monkey patch Path
from fastai.vision import *

from artifical_contrast.dicom import read_HU_array
from artifical_contrast.freqs import freqhist_bins, hist_scaled_img

DATA_PATH = 'unet2d/data/bc_and_segmentation_133'
WINDOWS = [-300, 300]

scans = []

bc_and_segmentation_path = Path(DATA_PATH)
patients_s = pd.Series(sorted(os.listdir(bc_and_segmentation_path)))

patients_s = patients_s[patients_s != 'P2B2']

# P2B2 is removed so we can just sample
for patient in patients_s.sample(1, random_state=42).tolist():
    print(f'Patient: {patient}')
    bc_dir = Path(bc_and_segmentation_path / patient / 'BC')
    scans += [str(p) for p in bc_dir.ls()]


sample_dcms = torch.stack(tuple([read_HU_array(fn) for fn in sorted(scans)]))

if WINDOWS:
    min_val, max_val = WINDOWS
    sample_dcms[sample_dcms < min_val] = min_val
    sample_dcms[sample_dcms < max_val] = max_val

freqs = freqhist_bins(sample_dcms)
np.save('freqs.npy', freqs.numpy())

scaled_samples = torch.stack(
    tuple([hist_scaled_img(scan, freq_bins=freqs) for scan in scans])
)
# for normalization
mean, std = scaled_samples.mean(), scaled_samples.std()
print(mean, std)
