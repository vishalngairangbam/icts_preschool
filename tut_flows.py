import os,sys 


from tqdm import tqdm 
import flows 
import jax 
import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import itertools 
import optax

import matplotlib.pyplot as plt 
from itertools import combinations
from jax_flows import real_nvp,invertible_sigmoid
#rng=jax.random.PRNGKey(989098) 
def transformation():
    def init_fn(rng,input_dim,output_dim):
        _mod=nn.Dense(output_dim)
        params=_mod.init(rng,jnp.ones(input_dim))
        def apply_fun(params,inp):
            return _mod.apply(params,inp) 
        return params,apply_fun
    return init_fn
#init_fn=transformation() 
#print (init_fn(rng,2,4))
#sys.exit() 


def plot_check(X_samples,X_true,name='check',num_samples=3000):
    fig,axes=plt.subplots(ncols=3,figsize=(30,10)) 
    for i,c in enumerate(combinations(range(3),2)):
        #print (i,c)
        axes[i].scatter(X_samples[:num_samples,c[0]],X_samples[:num_samples,c[1]],marker='x',c='r',label='Sampled') 
        axes[i].scatter(X_true[:num_samples,c[0]],X_true[:num_samples,c[1]],marker='o',c='g',label='True')
    axes[0].legend() 
    fig.savefig(name) 


rng=jax.random.PRNGKey(989098) 
inp_rng,data_rng,sample_rng=jax.random.split(rng,3) 
bijection = flows.Serial(
    *[flows.AffineCoupling(transformation()) for _ in range(10)]
)
prior = flows.Normal()

#init_fun = flows.Flow(bijection, prior)
init_fun=real_nvp(num_layers=3,with_scale=True)
#init_fun=invertible_sigmoid()
means=jnp.array([-2,1,3],dtype=jnp.float64)
pre_array=jnp.array([[3,1,2],[1,2,3],[2,3,1]],dtype=jnp.float64)  
covariance=(pre_array.T @ pre_array)#/jnp.linalg.det(pre_array) 
covariance=15*covariance/jnp.linalg.det(covariance) 
#print (covariance)
#sys.exit()  
X=jax.random.multivariate_normal(data_rng,means,covariance,shape=(100_000,)) 

print (len(means))
params, log_pdf, sample = init_fun(rng, input_dim=len(means))
X_samples=sample(sample_rng,params,100_000) 
print (X_samples.shape) 
plot_check(X_samples,X,name='before_train') 
def loss(params, inputs):
    return -log_pdf(params, inputs).mean()

@jax.jit
def step(i, params,solver_state, inputs):
    grad = jax.grad(loss)(params, inputs)
    updates, new_state = optimiser.update(grad, solver_state, params)

    return optax.apply_updates(params, updates),new_state

batch_size, itercount = 1000, itertools.count()

optimiser=optax.adam(0.001,nesterov=False)
solver_state=optimiser.init(params)
num_epochs=10
all_losses=np.zeros(num_epochs+1) 
best_params=None
best_loss=float('inf') 
best_epoch=None
for epoch in range(num_epochs):
    #jnp.random.shuffle(X)
    
    if epoch==0:
        total_loss=jax.jit(loss)(params,X)/len(X)  
        print ('Loss before training: ',total_loss) 
        all_losses[0]=total_loss.item() 
    for batch_index in tqdm(range(0, X.shape[0], batch_size)):
        params,solver_state = step(
            next(itercount),
            params,solver_state,
            X[batch_index:batch_index+batch_size]
        )
    
    total_loss=jax.jit(loss)(params,X)/len(X) 
    if total_loss <best_loss:
        best_params=params
        best_loss=total_loss 
        best_epoch=epoch
    print (f' Epoch: {epoch+1} Val loss: {total_loss}')
    all_losses[epoch+1]=total_loss.item() 
print (f'Best Epoch: {best_epoch+1} Best loss: {best_loss}') 

fig,axes=plt.subplots(figsize=(16,10))
axes.set_yscale('log') 
axes.plot(np.arange(len(all_losses)),all_losses) 
fig.savefig('flow_try') 

X_samples=sample(sample_rng,best_params,100_000) 
print (X_samples.shape) 
plot_check(X_samples,X,name='after_train') 













