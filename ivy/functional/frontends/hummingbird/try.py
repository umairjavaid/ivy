import ivy

class myivymodel(ivy.Module):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def forward(self, x):
        x = self.var1 * x
        x = self.var2 + x
        return x

def get_device(model):
    """
    Convenient function used to get the runtime device for the model.
    """
    assert issubclass(model.__class__, ivy.Module)

    device = None
    if len(list(model.parameters())) > 0:
        device = next(model.parameters()).device  # Assuming we are using a single device for all parameters

    return device

def main():
    mymodel = myivymodel(ivy.array(2),ivy.array(2))
    print(mymodel(ivy.array(2)))

if __name__ == "__main__":
    main()