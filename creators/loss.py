"""AudioSet loss factory."""

from functools import partial


def _create_audioset_loss(hparams):
    from loss_functions.audioset_loss import audioset_strong_loss

    return partial(
        audioset_strong_loss,
        class_positive_weight=hparams.class_positive_weight,
        balance_positive_negative=hparams.balance_positive_negative,
    )


def loss_creator(hparams):
    creators = {"audioset_strong": _create_audioset_loss}
    try:
        creator = creators[hparams.loss_type]
    except KeyError as error:
        raise ValueError(f"Unsupported loss_type: {hparams.loss_type}") from error
    return creator(hparams)
