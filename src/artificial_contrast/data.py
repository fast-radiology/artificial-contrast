import os
from pathlib import Path
import pandas as pd
from fastai.vision import get_transforms, SegmentationItemList


CODES = ['no_contrast', 'contrast']


def get_scans(data_path):
    scans = []

    patients = sorted(os.listdir(data_path))

    for patient in patients:
        bc_dir = Path(data_path / patient / 'BC')
        scans += [str(p) for p in bc_dir.ls()]

    return sorted(scans)


def get_patients(data_path):
    return sorted(os.listdir(data_path))


def get_y_fn(path):
    bc_path = str(path)
    label_path = bc_path.replace('BC', 'label')

    return label_path


def get_data(scans, home_path, validation_patients, bs, size=512, normalize_stats=None):
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
