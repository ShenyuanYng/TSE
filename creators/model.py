"""AudioSet model factory."""

"""
根据配置决定实例化那个模型，并把模型相关的超参数传给构造函数
"""
def _create_audioset_crnn(hparams):
    from nets.audioset_crnn import AudioSetCRNN

    return AudioSetCRNN(
        num_classes=hparams.num_classes,
        num_imics=hparams.num_imics,
        conv_channels=hparams.audioset_conv_channels,#[32,64,96]三个CNN块的输出通道
        temporal_hidden=hparams.audioset_temporal_hidden,#GRU每个方向的隐藏维度
        rnn_layers=hparams.audioset_rnn_layers,#GRU层数
        dropout=hparams.audioset_dropout,#CNN,GRU和融合层使用的dropout
    )


def model_creator(hparams):
    creators = {"AudioSetCRNN": _create_audioset_crnn}  #依旧是字典
    try:
        creator = creators[hparams.model_name]
    except KeyError as error:
        raise ValueError(f"Unsupported model_name: {hparams.model_name}") from error
    return creator(hparams)
