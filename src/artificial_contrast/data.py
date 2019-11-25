import os
from pathlib import Path

import pandas as pd

from fastai.vision import get_transforms, SegmentationItemList
from artificial_contrast.settings import TEST_SET_EXAMINATIONS


CODES = ['no_contrast', 'contrast']


def get_scans(data_path, test=False):
    scans = []

    patients = get_patients(data_path, test)
    for patient in patients:
        bc_dir = Path(data_path / patient / 'BC')
        scans += [str(p) for p in bc_dir.ls()]

    return sorted(scans)


def get_patients(data_path, test=False):
    if test:
        patients = TEST_SET_EXAMINATIONS
        assert len(patients) == 11
    else:
        patients = [
            patient
            for patient in os.listdir(data_path)
            if patient not in TEST_SET_EXAMINATIONS
        ]
        assert len(patients) == 120

    return sorted(patients)


def get_y_fn(path):
    bc_path = str(path)
    label_path = bc_path.replace('BC', 'label')

    return label_path


def get_data(
    scans, home_path, validation_patients, bs, size=512, normalize_stats=None, tfms=None
):
    if tfms is None:
        tfms = get_transforms(max_rotate=5.0, max_lighting=0, p_lighting=0, max_warp=0)

    data = (
        SegmentationItemList.from_df(pd.DataFrame(scans), home_path)
        .split_by_valid_func(
            lambda path: any([p in str(path) for p in validation_patients])
        )
        .label_from_func(get_y_fn, classes=CODES)
        .transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(normalize_stats)
    )

    return data
