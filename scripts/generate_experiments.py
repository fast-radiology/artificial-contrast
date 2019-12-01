from fast_radiology.seed import random_seed
from artificial_contrast.const import SEED

random_seed(SEED)

from typing import List, Dict
import numpy as np
import torch

# monkey patch Path from pathlib
from fastai.vision import *

from artificial_contrast.data import get_scans, get_patients
from artificial_contrast.dicom import read_HU_array
from artificial_contrast.freqs import (
    freqhist_bins,
    hist_scaled_img,
    remove_voxels_outside_window,
    get_freqs_array,
)
from artificial_contrast.const import (
    FOLDS_NAME,
    FREQS_LIMIT_WINDOWS,
    FREQS_NO_LIMIT_WINDOWS,
    FREQS,
    NORM_STATS,
    TRAIN_PATIENTS,
    VALIDATION_PATIENTS,
    WINDOWS,
)

from sklearn.model_selection import KFold


N_FOLDS = 10
STANDARD_WINDOWS = [-100, 300]
DATA_PATH = os.environ['DATA']

data_path = Path(DATA_PATH)
patients = np.array(get_patients(data_path))

assert len(patients) == 120
print('Number of patients: ', len(patients))


def get_freqs_method_dict(scans: List[str], windows: List[int]) -> Dict[str, object]:
    sample_dcms = torch.cat(
        tuple(
            [
                remove_voxels_outside_window(torch.tensor(read_HU_array(fn)), windows)
                for fn in sorted(scans)
            ]
        )
    )
    freqs = freqhist_bins(sample_dcms)
    method_result = {WINDOWS: windows, FREQS: freqs.tolist()}

    scaled_samples = torch.stack(
        tuple([get_freqs_array(scan, method_result) for scan in scans])
    )
    mean, std = scaled_samples.mean().item(), scaled_samples.std().item()

    method_result[NORM_STATS] = ([mean], [std])
    return method_result


folds = []
kfold = KFold(N_FOLDS, shuffle=True, random_state=SEED)
for train_index, val_index in kfold.split(patients):
    result = {}
    # TODO remove [:2] after confirming it's working
    train_patients = patients[train_index][:2]
    val_patients = patients[val_index]
    print(val_patients)

    result[TRAIN_PATIENTS] = train_patients.tolist()
    result[VALIDATION_PATIENTS] = val_patients.tolist()

    scans = get_scans(data_path, patients=train_patients)

    # freqs
    result[FREQS_NO_LIMIT_WINDOWS] = get_freqs_method_dict(scans, None)
    result[FREQS_LIMIT_WINDOWS] = get_freqs_method_dict(scans, STANDARD_WINDOWS)

    folds.append(result)
    print(result)


pd.DataFrame(folds).to_csv(FOLDS_NAME, encoding='utf-8', index=False)
