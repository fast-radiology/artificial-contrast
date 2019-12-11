import os
import argparse
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
    SIMPLE_WINDOW_SMALL,
    VALIDATION_PATIENTS,
)
from artificial_contrast.evaluate import evaluate_patients

fastai.vision.image.open_mask = open_dcm_mask
fastai.vision.data.open_mask = open_dcm_mask
open_mask = open_dcm_mask


# To explore later
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int)
# args = parser.parse_args()
# torch.cuda.set_device(args.local_rank)
# torch.distributed.init_process_group(backend='nccl', init_method='env://')


# Config

HOME_PATH = os.environ['HOME']
DATA_PATH = os.environ['DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']
FOLDS_PATH = os.environ['FOLDS']
data_path = Path(DATA_PATH)

EXPERIMENT_NAME = os.envrion['EXPERIMENT_NAME']

IMG_SIZE = 512
BS = 20


DCM_LOAD_FUNC = {
    SIMPLE_WINDOW_SMALL: simple_open_dcm_image_factory,
    SIMPLE_MULTIPLE_WINDOWS: simple_open_dcm_image_factory,
    FREQS_NO_LIMIT_WINDOWS: freqs_open_dcm_image_factory,
    FREQS_LIMIT_WINDOWS: freqs_open_dcm_image_factory,
}


# CV

scans = get_scans(data_path)
folds_df = pd.DataFrame(FOLDS_PATH)

results = []

for idx, fold in folds_df.iterrows():
    conf = json.loads(fold[EXPERIMENT_NAME])
    open_dcm_image_func = DCM_LOAD_FUNC[EXPERIMENT_NAME](conf)

    fastai.vision.image.open_image = open_dcm_image_func
    fastai.vision.data.open_image = open_dcm_image_func
    open_image = open_dcm_image_func

    validation_patients = fold[VALIDATION_PATIENTS]
    print('Validation patients: ', validation_patients)

    data = get_data(
        scans, HOME_PATH, validation_patients, normalize_stats=conf[NORM_STATS], bs=BS
    )
    learn = get_learner(data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)

    learn.fit_one_cycle(10, 1e-4)

    fold_results_df = evaluate_patients(learn, validation_patients, IMG_SIZE)
    result = {
        'fold': idx,
        'mean': fold_results_df[DICE_NAME].mean(),
        'std': fold_results_df[DICE_NAME].std(),
    }
    results.append(result)

    print(result)
    print(fold_results_df)

print(pd.DataFrame(results))
