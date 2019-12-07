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
from artificial_contrast.const import (
    FREQS_LIMIT_WINDOWS,
    FREQS_NO_LIMIT_WINDOWS,
    SIMPLE_MULTIPLE_WINDOWS,
    SIMPLE_WINDOW_SMALL,
)
from artificial_contrast.evaluate import evaluate_patients


# Config

HOME_PATH = os.environ['HOME']
DATA_PATH = os.environ['DATA']
TEST_DATA_PATH = os.environ['TEST_DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']
MODEL_NAME = os.environ['MODEL_NAME']
DCM_CONF_PATH = os.environ['DCM_CONF']

data_path = Path(DATA_PATH)
test_data_path = Path(TEST_DATA_PATH)
train_data_path = Path(TRA)

EXPERIMENT_NAME = os.envrion['EXPERIMENT_NAME']

IMG_SIZE = 512
BS = 20


DCM_LOAD_FUNC = {
    SIMPLE_WINDOW_SMALL: simple_open_dcm_image_factory,
    SIMPLE_MULTIPLE_WINDOWS: simple_open_dcm_image_factory,
    # FREQS_NO_LIMIT_WINDOWS: freqs_open_dcm_image_factory,
    # FREQS_LIMIT_WINDOWS: freqs_open_dcm_image_factory,
}


# TRAIN

with open(DCM_CONF_PATH) as json_file:
    conf = json.load(json_file)
open_dcm_image_func = DCM_LOAD_FUNC[EXPERIMENT_NAME](conf)
fastai.vision.image.open_image = open_dcm_image_func
fastai.vision.data.open_image = open_dcm_image_func
open_image = open_dcm_image_func

scans = get_scans(data_path)

validation_patients = []
print('Validation patients: ', validation_patients)

data = get_data(scans, HOME_PATH, validation_patients, bs=BS)
learn = get_learner(data, metrics=[], model_save_path=MODEL_SAVE_PATH)

learn.fit_one_cycle(20, 1e-4)
learn.save(MODEL_NAME)


# TEST (or maybe split to two scripts)

# Extra checks, remove after running successfully
del scans, data, data_path, validation_patients, learn

test_patients = get_patients(test_data_path)
print('Test patients: ', test_patients)

test_scans = get_scans(test_data_path, patients=test_patients)
test_data = get_data(test_scans, HOME_PATH, test_patients, bs=BS)
learn = get_learner(test_data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)
learn.load(MODEL_NAME)

eval_df = evaluate_patients(learn, test_patients, IMG_SIZE)

print(eval_df)
print(f"mean: {eval_df[DICE_NAME].mean()}, std: {eval_df[DICE_NAME].std()}")
