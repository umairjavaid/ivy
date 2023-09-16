import torch
import ivy
import jax
jax.config.update("jax_enable_x64", True)
import tensorflow as tf
import numpy as np
import haiku as hk

class myivymodel(torch.nn.Module):
    def __init__(self, var1, var2):
        super().__init__()
        self.var1 = var1
        self.var2 = var2

    def forward(self, x):
        x = self.var1 * x
        x = self.var2 + x
        return x



def main():
    mymodel = myivymodel(torch.tensor(2),torch.tensor(2)) 
    myjaxmodel = ivy.transpile(mymodel, source= "torch", to="jax")
    transformed_model = hk.transform_with_state(myjaxmodel)
    rng = jax.random.PRNGKey(0)
    x = jax.numpy.array([2])
    params, state = transformed_model.init(rng=rng, x=x)
    
    
    output, state = transformed_model.apply(params, state, x)
    print(output)

if __name__ == "__main__":
    main()