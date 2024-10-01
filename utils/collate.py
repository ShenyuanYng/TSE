import collections.abc as container_abcs
import torch
import torch.nn.utils.rnn as rnn
import numpy as np


def collate_seq(batch, pad_back_or_front=0):
    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__name__ == 'str':
        return [b for b in batch]
    elif elem_type.__name__ == 'ndarray':
        lengths = torch.tensor([len(b) for b in batch])
        if pad_back_or_front == 0 or len(batch) == 1:
            # padding back
            pad_features = rnn.pad_sequence([torch.tensor(b, dtype=torch.float32) for b in batch],
                                            batch_first=True)
        elif pad_back_or_front == 1:
            # padding front
            pad_features = torch.flip(
                    rnn.pad_sequence(
                        [torch.flip(torch.tensor(b, dtype=torch.float32),dims=[0,]) for b in batch],
                        batch_first=True),
                    dims=[1,],
                    )
        else:
            raise
        return pad_features, lengths
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_seq(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_seq([d[key] for d in batch]) for key in elem}
    else:
        # not collate
        return [b for b in batch]
