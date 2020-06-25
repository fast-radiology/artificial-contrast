from fastai.vision import unet_learner, models, Learner

from fast_radiology.losses import generalized_dice_loss
from fast_radiology.models.unet import ClassicUNet
from artificial_contrast.const import (
    FREQS_LIMIT_WINDOWS,
    FREQS_NO_LIMIT_WINDOWS,
    SIMPLE_MULTIPLE_WINDOWS,
    SIMPLE_WINDOW_1CHANNEL_CLASSIC_UNET,
    SIMPLE_WINDOW_3CHANNEL_CLASSIC_UNET,
    SIMPLE_WINDOW_SMALL,
)


def get_learner(
    data, metrics=None, model_save_path='models', loss_func=None, pretrained=True, experiment_name=None
):
    if metrics is None:
        metrics = []
    if loss_func is None:
        loss_func = generalized_dice_loss

    if experiment_name is None or experiment_name in [
        FREQS_LIMIT_WINDOWS, FREQS_NO_LIMIT_WINDOWS, SIMPLE_MULTIPLE_WINDOWS, SIMPLE_WINDOW_SMALL
    ]:
        return unet_learner(
            data,
            models.resnet18,
            metrics=metrics,
            self_attention=True,
            loss_func=loss_func,
            path=model_save_path,
            pretrained=pretrained,
        )
    elif experiment_name in [
        SIMPLE_WINDOW_1CHANNEL_CLASSIC_UNET, SIMPLE_WINDOW_3CHANNEL_CLASSIC_UNET
    ]:
        if experiment_name == SIMPLE_WINDOW_1CHANNEL_CLASSIC_UNET:
            n_channels = 1
        elif experiment_name == SIMPLE_WINDOW_3CHANNEL_CLASSIC_UNET:
            n_channels = 3

        return Learner(
            data,
            ClassicUNet(n_channels=n_channels, n_classes=2),
            metrics=metrics,
            loss_func=loss_func,
            path=model_save_path,
        )
    else:
        raise RuntimeError('Incorrect experiment_name')
