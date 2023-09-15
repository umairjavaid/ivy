import ivy

class myivymodel(ivy.Module):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        super().__init__()

    def _forward(self, x):
        x = self.var1 * x
        x = self.var2 + x
        return x

def get_device(model):
    return model._device

def main():
    mymodel = myivymodel(ivy.array(2),ivy.array(2))
    print(mymodel(ivy.array(2)))
    print(get_device(mymodel))

if __name__ == "__main__":
    main()