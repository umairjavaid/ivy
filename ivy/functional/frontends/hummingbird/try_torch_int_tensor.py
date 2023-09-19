import torch 
import ivy

class mytorchmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.var = torch.tensor([1,2]).to(torch.int64)
        self.var1 = torch.tensor([3,4]).to(torch.float64)
    
    def forward(self):
        with torch.no_grad():
            return self.var


model = mytorchmodel()
print(model)
print(model.var.requires_grad)
print(model.var1.requires_grad)
ivymodel = ivy.unify(model, source="torch")
print(ivymodel)
print(ivymodel.v)
ivymodel.var = ivy.array([5,6])
print(ivymodel.var)
print(ivymodel._forward())
