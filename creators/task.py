"""Task adapter between the shared Lightning loop and AudioSet Model 1."""


"""
只负责选择任务类型，并把框架提供的数据转交给模型
"""
class AudioSetStrongTask:
    def forward(self, model, model_input, target_dict, current_epoch):
        return model(
            model_input,
            target_dict=target_dict,
            current_epoch=current_epoch,
        )
    """
    model:已经创建好的模型
    model_input：模型输入字典
    target_dict：强标签等训练目标组成的字典
    current_epoch：当前训练轮数
    """


def task_creator(hparams):
    task_type = getattr(hparams, "task_type", hparams.loss_type)#读取任务类型
    creators = {"audioset_strong": AudioSetStrongTask}#creators是字典，通过键（audioset_strong）值（AudioSetStrongTask）对
    try:
        return creators[task_type]() #返回一个任务对象
    except KeyError as error:
        raise ValueError(f"Unsupported task_type: {task_type}") from error
