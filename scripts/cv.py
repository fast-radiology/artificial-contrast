import os

# import argparse
import json

from fast_radiology.seed import random_seed
from artificial_contrast.const import SEED

random_seed(SEED)

import torch
import fastai
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.distributed import *

from fast_radiology.metrics import dice as dice3D
from artificial_contrast.freqs import (
    open_dcm_img_factory as freqs_open_dcm_image_factory,
)
from artificial_contrast.dicom import (
    open_dcm_image_factory as simple_open_dcm_image_factory,
    open_dcm_mask,
)
from artificial_contrast.data import get_scans, get_data
from artificial_contrast.learner import get_learner
from artificial_contrast.const import (
    DICE_NAME,
    FREQS_LIMIT_WINDOWS,
    FREQS_NO_LIMIT_WINDOWS,
    NORM_STATS,
    SIMPLE_MULTIPLE_WINDOWS,
    SIMPLE_WINDOW_1CHANNEL_CLASSIC_UNET,
    SIMPLE_WINDOW_3CHANNEL_CLASSIC_UNET,
    SIMPLE_WINDOW_SMALL,
    VALIDATION_PATIENTS,
)
from artificial_contrast.evaluate import evaluate_patients

fastai.vision.image.open_mask = open_dcm_mask
fastai.vision.data.open_mask = open_dcm_mask
open_mask = open_dcm_mask


# Config

HOME_PATH = os.environ['REPO']
DATA_PATH = os.environ['DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']
RESULTS_PATH = os.environ['RESULTS']
FOLDS_PATH = os.environ['FOLDS']
data_path = Path(DATA_PATH)

EXPERIMENT_NAME = os.environ['EXPERIMENT_NAME']
print(EXPERIMENT_NAME)

IMG_SIZE = 512
BS = int(os.environ.get('BS', 20))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 16))


DCM_LOAD_FUNC = {
    SIMPLE_WINDOW_SMALL: simple_open_dcm_image_factory,
    SIMPLE_MULTIPLE_WINDOWS: simple_open_dcm_image_factory,
    FREQS_NO_LIMIT_WINDOWS: freqs_open_dcm_image_factory,
    FREQS_LIMIT_WINDOWS: freqs_open_dcm_image_factory,
    SIMPLE_WINDOW_1CHANNEL_CLASSIC_UNET: simple_open_dcm_image_factory,
    SIMPLE_WINDOW_3CHANNEL_CLASSIC_UNET: simple_open_dcm_image_factory,
}


# CV

scans = get_scans(data_path)
folds_df = pd.read_csv(FOLDS_PATH, encoding='utf-8')

results = []
by_patient_results = []

for idx, fold in folds_df.iterrows():
    conf = json.loads(fold[EXPERIMENT_NAME])
    open_dcm_image_func = DCM_LOAD_FUNC[EXPERIMENT_NAME](conf)

    fastai.vision.image.open_image = open_dcm_image_func
    fastai.vision.data.open_image = open_dcm_image_func
    open_image = open_dcm_image_func

    validation_patients = json.loads(fold[VALIDATION_PATIENTS])
    print('Validation patients: ', validation_patients)

    data = get_data(
        scans, HOME_PATH, validation_patients, normalize_stats=conf[NORM_STATS], bs=BS
    )
    learn = get_learner(
        data,
        metrics=[dice],
        model_save_path=MODEL_SAVE_PATH,
        experiment_name=EXPERIMENT_NAME,
    )

    learn.unfreeze()
    learn.fit_one_cycle(NUM_EPOCHS, 1e-4)

    fold_results_df = evaluate_patients(learn, validation_patients, IMG_SIZE)
    result = {
        'fold': idx,
        'mean': fold_results_df[DICE_NAME].mean(),
        'std': fold_results_df[DICE_NAME].std(),
    }
    results.append(result)
    by_patient_results.append(fold_results_df)

    print(result)
    print(fold_results_df)

df = pd.DataFrame(results)
print(df)
df.to_csv(
    os.path.join(RESULTS_PATH, f"{EXPERIMENT_NAME}_result.csv"),
    index=False,
    encoding='utf-8',
)

by_patient_result_df = pd.concat(by_patient_results)
print(by_patient_result_df)
by_patient_result_df.to_csv(
    os.path.join(RESULTS_PATH, f"{EXPERIMENT_NAME}_by_patient_result.csv"),
    index=False,
    encoding='utf-8',
)
