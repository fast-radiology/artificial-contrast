import os
import argparse
import json

from fast_radiology.seed import random_seed
from artificial_contrast.settings import SEED

random_seed(SEED)

import torch
import fastai
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.distributed import *

from fast_radiology.metrics import dice as dice3D
from artificial_contrast.dicom import (
    open_dcm_image_factory as simple_open_dcm_image_factory,
    open_dcm_mask,
)
from artificial_contrast.data import get_scans, get_data
from artificial_contrast.learner import get_learner


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
    'simple_window_-100_300': simple_open_dcm_image_factory,
    'simple_multiple_windows': simple_open_dcm_image_factory,
    # 'freqs_no_limit_window': freqs_open_dcm_image_factory,
    # 'freqs_window_-100_300': freqs_open_dcm_image_factory,
}


# CV

scans = get_scans(data_path)
folds_df = pd.DataFrame(FOLDS_PATH)

for idx, fold in folds_df.iterrows():
    conf = json.loads(fold[EXPERIMENT_NAME])
    open_dcm_image_func = DCM_LOAD_FUNC(conf)

    fastai.vision.image.open_image = open_dcm_image_func
    fastai.vision.data.open_image = open_dcm_image_func
    open_image = open_dcm_image_func

    validation_patients = fold['validation_patients']
    print('Validation patients: ', validation_patients)

    data = get_data(scans, HOME_PATH, validation_patients, bs=BS)
    learn = get_learner(data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)
    learn = learn.to_distributed(args.local_rank)

    learn.fit_one_cycle(10, 1e-4)

    preds, targets = learn.get_preds()

    preds_df = pd.DataFrame(
        {
            'preds': [
                preds[i].argmax(0).view(1, IMG_SIZE, IMG_SIZE).int().numpy()
                for i in range(len(preds))
            ],
            'targets': [
                targets[i].view(1, IMG_SIZE, IMG_SIZE).int().numpy()
                for i in range(len(targets))
            ],
            'path': learn.data.valid_ds.items,
        }
    )

    fold_results = []

    for val_patient in validation_patients:
        val_pred_3d = torch.tensor(
            preds_df[preds_df['path'].str.contains(val_patient)]
            .sort_values('path')['preds']
            .to_list()
        )
        val_pred_3d = val_pred_3d.view(-1, IMG_SIZE, IMG_SIZE)
        val_target_3d = torch.tensor(
            preds_df[preds_df['path'].str.contains(val_patient)]
            .sort_values('path')['targets']
            .to_list()
        )
        val_target_3d = val_target_3d.view(-1, IMG_SIZE, IMG_SIZE)

        patient_dice = dice3D(val_pred_3d, val_target_3d)

        fold_results.append({'patient': val_patient, 'dice': patient_dice.item()})

    fold_results_df = pd.DataFrame(fold_results)
    print(fold_results_df)
    print(
        f"mean: {fold_results_df['dice'].mean()}, std: {fold_results_df['dice'].std()}"
    )
