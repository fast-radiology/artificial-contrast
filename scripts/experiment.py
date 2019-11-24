import os
import argparse

from fast_radiology.seed import random_seed
from artificial_contrast.settings import SEED

random_seed(SEED)

import torch
import fastai
from fastai.vision import *
from fastai.distributed import *

from fast_radiology.metrics import dice
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


# Experiment

scans = get_scans(data_path)
patients = get_patients(data_path)
validation_patients = [f'P{i}B{i}' for i in range(10) if f'P{i}B{i}' in patients]

print('Number of patients: ', len(patients))
print('Validation patients: ', validation_patients)

data = get_data(scans, HOME_PATH, validation_patients, bs=BS)
learn = get_learner(data, metrics=[dice], model_save_path=MODEL_SAVE_PATH)
learn = learn.to_distributed(args.local_rank)

learn.fit_one_cycle(20, 1e-4)

# learn.save('my_model_1')
