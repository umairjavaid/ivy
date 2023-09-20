import torch
import ivy
import jax
jax.config.update("jax_enable_x64", True)
rng_key = jax.random.PRNGKey(42)

ivy.set_backend("jax")
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
    myjaxmodel = ivy.unify(mymodel, source= "torch")

    def _forward(**kwargs):
        module = myjaxmodel()
        return module(**kwargs)
    
    jax_forward = hk.transform(_forward)

    inputs_jax = {"x": jax.numpy.array([2])}
    
    params = jax_forward.init(rng=rng_key, **inputs_jax)
    jit_apply = jax.jit(jax_forward.apply)
    out = jit_apply(params, None, **inputs_jax)
    print(out)

if __name__ == "__main__":
    main()