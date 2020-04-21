import os
import json
from random import sample

from fast_radiology.seed import random_seed
from artificial_contrast.const import SEED

random_seed(SEED)

import torch
import fastai
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.distributed import *

from artificial_contrast.dicom import open_dcm_mask
from artificial_contrast.dicom import (
    open_dcm_image_factory as simple_open_dcm_image_factory,
)
from artificial_contrast.data import get_scans, get_data, get_patients
from artificial_contrast.learner import get_learner
from artificial_contrast.const import DICE_NAME, NORM_STATS
from artificial_contrast.evaluate import evaluate_patients

fastai.vision.image.open_mask = open_dcm_mask
fastai.vision.data.open_mask = open_dcm_mask
open_mask = open_dcm_mask

# Config

HOME_PATH = os.environ['REPO']
DATA_PATH = os.environ['DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']
RESULTS_PATH = os.environ['RESULTS']

data_path = Path(DATA_PATH)

SAMPLE_SIZES = [20, 40, 60, 80, 100]
SAMPLING_ROUNDS = 5

IMG_SIZE = 512
BS = 20

DCM_CONF = json.loads(os.environ['DCM_CONF'])
VALIDATION_PATIENTS = json.loads(os.environ['VALIDATION_PATIENTS'])

open_dcm_image_func = simple_open_dcm_image_factory(DCM_CONF)
fastai.vision.image.open_image = open_dcm_image_func
fastai.vision.data.open_image = open_dcm_image_func
open_image = open_dcm_image_func

# EXPERIMENT

patients = get_patients(data_path)
training_patients = [p for p in patients if p not in VALIDATION_PATIENTS]
print(
    f'patients: {len(patients)}, training: {len(training_patients)}, validation: {len(VALIDATION_PATIENTS)}'
)
print(f'validation patients: {VALIDATION_PATIENTS}')

results = []

for sample_size in SAMPLE_SIZES:
    print(f'Sample size: {sample_size}')

    for sampling_round in range(SAMPLING_ROUNDS):
        sampled_patients = sample(training_patients, sample_size)
        print(f'{sampling_round} sampled patients: {sampled_patients}')

        scans = get_scans(data_path, patients=sampled_patients + VALIDATION_PATIENTS)

        data = get_data(
            scans,
            HOME_PATH,
            VALIDATION_PATIENTS,
            normalize_stats=DCM_CONF[NORM_STATS],
            bs=BS,
        )
        learn = get_learner(data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)

        learn.unfreeze()
        learn.fit_one_cycle(16, 1e-4)

        round_eval_df = evaluate_patients(learn, VALIDATION_PATIENTS, IMG_SIZE)

        result = {
            'sample_size': sample_size,
            'sampling_round': sampling_round,
            'mean': round_eval_df[DICE_NAME].mean(),
            'std': round_eval_df[DICE_NAME].std(),
            'min': round_eval_df[DICE_NAME].min(),
            'max': round_eval_df[DICE_NAME].max(),
        }
        results.append(result)

        print(result)
        print(round_eval_df)


results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv(
    os.path.join(RESULTS_PATH, f"samplesize_effect_result.csv"),
    index=False,
    encoding='utf-8',
)
