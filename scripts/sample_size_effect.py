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
from sklearn.model_selection import KFold

from artificial_contrast.dicom import open_dcm_mask
from artificial_contrast.freqs import (
    open_dcm_img_factory as freqs_open_dcm_image_factory,
)
from artificial_contrast.data import get_scans, get_data, get_patients
from artificial_contrast.learner import get_learner
from artificial_contrast.const import DICE_NAME, NORM_STATS
from artificial_contrast.evaluate import evaluate_patients

fastai.vision.image.open_mask = open_dcm_mask
fastai.vision.data.open_mask = open_dcm_mask
open_mask = open_dcm_mask

# Config

HOME_PATH = os.environ['HOME']
DATA_PATH = os.environ['DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']

data_path = Path(DATA_PATH)

SAMPLE_SIZES = [30, 60, 90, 120]
SAMPLING_ROUNDS = 5
PER_SAMPLE_FOLDS = 5

IMG_SIZE = 512
BS = 20

DCM_CONF = json.loads(os.environ['DCM_CONF'])

open_dcm_image_func = freqs_open_dcm_image_factory(DCM_CONF)
fastai.vision.image.open_image = open_dcm_image_func
fastai.vision.data.open_image = open_dcm_image_func
open_image = open_dcm_image_func

# EXPERIMENT

patients = get_patients(data_path)

for sample_size in SAMPLE_SIZES:
    print(f'Sample size: {sample_size}')

    # Or maybe do K-fold here too..
    for i in range(SAMPLING_ROUNDS):

        sampled_patients = sample(patients, sample_size)
        print(f'{i} sampled patients: {sampled_patients}')
        scans = get_scans(data_path, patients=sampled_patients)

        kfold = KFold(PER_SAMPLE_FOLDS, shuffle=True, random_state=SEED)
        for train_index, val_index in kfold.split(sampled_patients):
            validation_patients = sampled_patients[val_index]
            print('Validation patients: ', validation_patients)

            data = get_data(
                scans,
                HOME_PATH,
                validation_patients,
                normalize_stats=conf[NORM_STATS],
                bs=BS,
            )
            learn = get_learner(data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)

            learn.fit_one_cycle(10, 1e-4)

            eval_df = evaluate_patients(learn, validation_patients, IMG_SIZE)
            print(eval_df)
            # TODO: aggregate all results into DF
            print(f"mean: {eval_df[DICE_NAME].mean()}, std: {eval_df[DICE_NAME].std()}")
