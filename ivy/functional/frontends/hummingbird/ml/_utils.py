import ivy
#import torch
import numpy as np
from packaging.version import parse, Version

def ivy_installed():
    """
    Checks that *ivy* is available.
    """
    try:
        import ivy
        assert parse(ivy.__version__) > Version("1.0.0"), "Please install torch >1.0.0"

        return True
    except ImportError:
        return False

def get_device(model):
    """
    Convenient function used to get the runtime device for the model.
    """
    assert issubclass(model.__class__, torch.nn.Module)

    device = None
    if len(list(model.parameters())) > 0:
        device = next(model.parameters()).device  # Assuming we are using a single device for all parameters

    return device

def from_strings_to_ints(input, max_string_length):
    """
    Utility function used to transform string inputs into a numerical representation.
    """
    shape = list(input.shape)
    shape.append(max_string_length // 4)
    return np.array(input, dtype="|S" + str(max_string_length)).view(np.int32).reshape(shape)

class _Constants(object):
    """
    Class enabling the proper definition of constants.
    """

    def __init__(self, constants, other_constants=None):
        for constant in dir(constants):
            if constant.isupper():
                setattr(self, constant, getattr(constants, constant))
        for constant in dir(other_constants):
            if constant.isupper():
                setattr(self, constant, getattr(other_constants, constant))

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise ConstantError("Overwriting a constant is not allowed {}".format(name))
        self.__dict__[name] = value