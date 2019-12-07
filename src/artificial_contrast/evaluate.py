import torch
import pandas as pd
from fast_radiology.metrics import dice as dice3D

from artificial_contrast.const import (
    DICE_NAME,
    PATH_NAME,
    PATIENT_NAME,
    PREDICTIONS_NAME,
    TARGETS_NAME,
)


def evaluate_patients(learn, patients, img_size)
    results = []

    preds, targets = learn.get_preds()

    preds_df = pd.DataFrame(
        {
            PREDICTIONS_NAME: [
                preds[i].argmax(0).view(1, img_size, img_size).int().numpy()
                for i in range(len(preds))
            ],
            TARGETS_NAME: [
                targets[i].view(1, img_size, img_size).int().numpy()
                for i in range(len(targets))
            ],
            PATH_NAME: learn.data.valid_ds.items,
        }
    )

    for patient in patients:
        pred_3d = torch.tensor(
            preds_df[preds_df[PATH_NAME].str.contains(patient)]
            .sort_values(PATH_NAME)[PREDICTIONS_NAME]
            .to_list()
        )
        pred_3d = pred_3d.view(-1, img_size, img_size)
        target_3d = torch.tensor(
            preds_df[preds_df[PATH_NAME].str.contains(patient)]
            .sort_values(PATH_NAME)[TARGETS_NAME]
            .to_list()
        )
        target_3d = target_3d.view(-1, img_size, img_size)

        patient_dice = dice3D(pred_3d, target_3d)

        results.append({PATIENT_NAME: patient, DICE_NAME: patient_dice.item()})

    return pd.DataFrame(results)