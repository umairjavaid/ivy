import torch
import ivy

class mytorchmodel(torch.nn.Module):
    def __init__(self, var1, var2):
        super().__init__()
        self.var1 = var1
        self.var2 = var2

    def forward(self, x):
        x = self.var1 * x
        x = self.var2 + x
        return x


def main():
    myivymodel = ivy.unify(mytorchmodel(torch.tensor(2), torch.tensor(2)), source="torch")
    print(myivymodel._forward(ivy.array([2])))
    

if __name__ == "__main__":
    main()