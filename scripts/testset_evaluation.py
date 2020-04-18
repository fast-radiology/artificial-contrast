import os
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

from artificial_contrast.dicom import (
    open_dcm_image_factory as simple_open_dcm_image_factory,
    open_dcm_mask,
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
TEST_DATA_PATH = os.environ['TEST_DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']
MODEL_NAME = os.environ['MODEL_NAME']
RESULTS_PATH = os.environ['RESULTS']

data_path = Path(DATA_PATH)
test_data_path = Path(TEST_DATA_PATH)

IMG_SIZE = 512
BS = int(os.environ.get('BS', 20))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 16))

DCM_CONF = json.loads(os.environ['DCM_CONF'])

open_dcm_image_func = simple_open_dcm_image_factory(DCM_CONF)
fastai.vision.image.open_image = open_dcm_image_func
fastai.vision.data.open_image = open_dcm_image_func
open_image = open_dcm_image_func

# TRAIN

scans = get_scans(data_path)

validation_patients = []
print('Validation patients: ', validation_patients)

data = get_data(
    scans, HOME_PATH, validation_patients, normalize_stats=DCM_CONF[NORM_STATS], bs=BS
)
learn = get_learner(data, metrics=[], model_save_path=MODEL_SAVE_PATH)

learn.unfreeze()
learn.fit_one_cycle(NUM_EPOCHS, 1e-4)
learn.save(MODEL_NAME)


# TEST (or maybe split to two scripts)

# Extra checks, remove after running successfully
del data, data_path, validation_patients, learn

test_patients = get_patients(test_data_path)
print('Test patients: ', test_patients)

test_scans = get_scans(test_data_path, patients=test_patients)
test_data = get_data(
    scans[:10] + test_scans,
    HOME_PATH,
    test_patients,
    normalize_stats=DCM_CONF[NORM_STATS],
    bs=BS,
)
learn = get_learner(test_data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)
learn.load(MODEL_NAME)

eval_df = evaluate_patients(learn, test_patients, IMG_SIZE)

print(eval_df.describe())
print(eval_df)
eval_df.to_csv(
    os.path.join(RESULTS_PATH, f"{MODEL_NAME}_testset_result.csv"),
    index=False,
    encoding='utf-8',
)
