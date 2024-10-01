import itertools


def perms(num_sources):
    return list(itertools.permutations(range(num_sources)))

class LossFuncWrapper(object):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*args, *self.args, **kwargs, **self.kwargs)
