import flax.linen as nn
import flows
import jax.numpy as jnp
import os,sys 
import jax

import scipy
import numpy as np






class SingleTransform(nn.Module):
    output_dim:int=None 
    hidden_dims: tuple=(32,32,32) 
    @nn.compact
    def __call__(self,x):
        x=nn.Dense(self.hidden_dims[0])(x)
        x=nn.relu(x)
        x=nn.Dense(self.hidden_dims[1])(x)
        x=nn.relu(x)
        x=nn.Dense(self.hidden_dims[2])(x)
        x=nn.relu(x)
        x=nn.Dense(self.output_dim)(x)
        return x#nn.relu(x) 


def transformation():
    def init_fn(rng,input_dim,output_dim):
        _mod=SingleTransform(output_dim) 
        params=_mod.init(rng,jnp.ones(input_dim))
        def apply_fun(params,inp):
            return _mod.apply(params,inp) 
        return params,apply_fun
    return init_fn

#def mlp_transformation(hidden_nodes):
    




def real_nvp(num_layers=5,with_scale=False,prior=flows.Normal(),**kwargs ):
    if with_scale:
        bijection=flows.Serial(*[flows.AffineCouplingSplit(transformation(), transformation()) for _ in range(num_layers)]) 
    else:
        bijection = flows.Serial(
        *[flows.AffineCoupling(transformation()) for _ in range(num_layers)]
          )

    init_fn = flows.Flow(bijection, prior)
    return init_fn 
    
def invertible_sigmoid(num_layers=3,batch_norm=False,prior=flows.Normal(),**kwargs ):
    layers=[]
    for _ in range(num_layers):
        if batch_norm:
            layers += [flows.InvertibleLinear(),flows.BatchNorm(),flows.Sigmoid()]
        else:
            layers += [flows.InvertibleLinear(),flows.Sigmoid()]    
    bijection=flows.Serial(*layers) 
    init_fn = flows.Flow(bijection, prior)
    return init_fn 




if __name__=='__main__':
    init_fn=multivariate_normal() 
    
    init_rng,rng,s_rng=jax.random.split(jax.random.PRNGKey(1029120),3)  
    
    params,log_pdf,sample=init_fn(init_rng,None,mean=jnp.array([0,0]),covariance=jnp.array([[1,0],[0,1]])) 
    s=sample(rng,params,10)
    
    print (s) 
    p=log_pdf(params,s) 
    print (s.shape,p.shape,p) 
    import flows 
    s_init_fn = flows.Normal()
    
    s_params,s_log_pdf,s_sample=s_init_fn(init_rng,2) 
    n_s=s_sample(s_rng,s_params,10)
    print (n_s,n_s.shape,s_log_pdf(s_params,s)) 
    print (s_log_pdf(s_params,n_s),log_pdf(params,n_s)) 





    
    
