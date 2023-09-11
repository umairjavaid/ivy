import ivy.functional.frontends.torch as torch


def get_device(model):
    """
    Convenient function used to get the runtime device for the model.
    """
    assert issubclass(model.__class__, torch.nn.Module)

    device = None
    if len(list(model.parameters())) > 0:
        device = next(model.parameters()).device  # Assuming we are using a single device for all parameters

    return device