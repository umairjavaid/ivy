import ivy
import torch


class torchmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.indices = torch.nn.Parameter(torch.tensor([1,2,3]).to(torch.int64), requires_grad=False)
    
    def forward(self):
        return self.indices
    

model = torchmodel()
print(model.forward())

model = ivy.unify(model, source="torch")
print(model._forward())
