# Config
# Set the data paths manually

HOME_PATH = ''
DATA_PATH = '/training_data/'
TEST_DATA_PATH = '/test_data/'
MODEL_SAVE_PATH = '/trained_models/'
RESULTS_PATH = '/results/'

# Set the model name for saving after training
MODEL_NAME = 'trained_model'

# Model parameters
IMG_SIZE = 512
BS = 20
NORM_STATS = [[0.191989466547966, 0.1603623628616333, 0.02605995163321495], [0.3100860118865967, 0.2717258334159851, 0.1233396977186203]]

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

def open_dcm_image_factory(conf):
    def open_dcm_image(file_name, *args, **kwargs) -> Image:
        arr = read_HU_array(file_name)

        windowed_arrays = []
        windows = [[-40, 120], [-100, 300], [300, 2000]]
        for window_min, window_max in windows:
            array = np.clip(arr, a_min=window_min, a_max=window_max)
            array = _normalize(array, window_min, window_max)

            windowed_arrays.append(array)

        final_array = np.dstack(windowed_arrays)
        return Image(pil2tensor(final_array, np.float32).div_(255))

    return open_dcm_image


data_path = Path(DATA_PATH)
test_data_path = Path(TEST_DATA_PATH)

# DCM_CONF = json.loads(os.environ['DCM_CONF'])

# open_dcm_image_func = simple_open_dcm_image_factory(DCM_CONF)
open_dcm_image_func = open_dcm_image_factory
fastai.vision.image.open_image = open_dcm_image_func
fastai.vision.data.open_image = open_dcm_image_func
open_image = open_dcm_image_func


scans = get_scans(data_path)
data = get_data(
    scans, HOME_PATH, validation_patients, normalize_stats=NORM_STATS, bs=BS
)

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
