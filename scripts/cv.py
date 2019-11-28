import os
import argparse

from fast_radiology.seed import random_seed

SEED = 42
random_seed(SEED)

import torch
import fastai
import numpy as np
import pandas as pd
from fastai.vision import *
from fastai.distributed import *
from sklearn.model_selection import KFold

from fast_radiology.metrics import dice as dice3D
from artificial_contrast.dicom import open_dcm_image, open_dcm_mask
from artificial_contrast.data import get_scans, get_patients, get_data
from artificial_contrast.learner import get_learner

fastai.vision.image.open_image = open_dcm_image
fastai.vision.image.open_mask = open_dcm_mask
fastai.vision.data.open_image = open_dcm_image
fastai.vision.data.open_mask = open_dcm_mask
open_image = open_dcm_image
open_mask = open_dcm_mask


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')


# Config

HOME_PATH = os.environ['HOME']
DATA_PATH = os.environ['DATA']
MODEL_SAVE_PATH = os.environ['MODEL_SAVE']
data_path = Path(DATA_PATH)

IMG_SIZE = 512
BS = 20


# CV
N_FOLDS = 10

scans = get_scans(data_path)
patients = get_patients(data_path)
print('Number of patients: ', len(patients))

kfold = KFold(N_FOLDS, shuffle=True, random_state=SEED)

for train_index, val_index in kfold.split(patients):
    validation_patients = list(np.array(patients)[val_index])
    print('Validation patients: ', validation_patients)

    data = get_data(scans, HOME_PATH, validation_patients, bs=BS)
    learn = get_learner(data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)
    learn = learn.to_distributed(args.local_rank)

    learn.fit_one_cycle(10, 1e-4)

    preds, targets = learn.get_preds()

    preds_df = pd.DataFrame(
        {
            'preds': [
                preds[i].argmax(0).view(1, size, size).int().numpy()
                for i in range(len(preds))
            ],
            'targets': [
                targets[i].view(1, size, size).int().numpy()
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
        val_pred_3d = val_pred_3d.view(-1, size, size)
        val_target_3d = torch.tensor(
            preds_df[preds_df['path'].str.contains(val_patient)]
            .sort_values('path')['targets']
            .to_list()
        )
        val_target_3d = val_target_3d.view(-1, size, size)

        patient_dice = dice3D(val_pred_3d, val_target_3d)

        fold_results.append({'patient': val_patient, 'dice': patient_dice})

    fold_results_df = pd.DataFrame(fold_results)
    print(fold_results_df)
    print(
        f"mean: {fold_results_df['dice'].mean()}, std: {fold_results_df['dice'].std()}"
    )
