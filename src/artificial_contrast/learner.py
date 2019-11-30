from fastai.vision import unet_learner, models

from fast_radiology.losses import generalized_dice_loss


def get_learner(
    data, metrics=None, model_save_path='models', loss_func=None, pretrained=True
):
    if metrics is None:
        metrics = []
    if loss_func is None:
        loss_func = generalized_dice_loss

    learner = unet_learner(
        data,
        models.resnet18,
        metrics=metrics,
        self_attention=True,
        loss_func=loss_func,
        path=model_save_path,
        pretrained=pretrained,
    )
    return learner
