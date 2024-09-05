#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:28:47 2024

@author: annabel
"""
from flax import linen as nn
import jax
from jax import numpy as jnp
from typing import Callable


class FcnnReg(nn.Module):
    """
    Simple dense model to use with one-hot encoded, aligned sequences
    Implements GLMs
    
    Can also use this to fit linear model; just pass empty list to 
      dense_size_lst
      
    Initialize with:
        dense_size_lst: for each intermediate layer, what should the output size be?
        activation_fn: which activation to apply between dense layers
        expOut: whether or not to apply final exp() activation function
    
    Training outputs:
        1: fitness coeff alone (for correlation)
        2: c * fitness coeff (for loss function)
    """
    
    dense_size_lst: list
    activation_fn: Callable
    expOut: bool
    
    
    @nn.compact
    def __call__(self, seq_c, training: bool, store_interms: bool):
        ### unpack sequences
        seq = seq_c[:, :-1] #(B, featSize)
        c = seq_c[:, -1][:,None] #(B, 1)
        
        
        ### start model here
        # repeat (dense -> activation)
        for layer_id,out_size in enumerate(self.dense_size_lst):
            seq = nn.Dense(features=out_size, use_bias=True)(seq)
            seq = self.activation_fn(seq)
            
            if store_interms:
                self.sow_histograms_scalars(mat = seq,
                                label = f"{self.name}/dense layer {layer_id}",
                                which = ['scalars'])
        
        # final projection to scores
        fit_coeff = nn.Dense(features=1, use_bias=True)(seq)
        
        # sow outputs
        if store_interms:
            self.sow_histograms_scalars(mat = fit_coeff,
                                        label = f"{self.name}/fit_coeff (without C)",
                                        which = ['hists','scalars'])
        
        # apply exp_out if desired
        if self.expOut:
            fit_coeff = jnp.exp(fit_coeff) #(B, 1)
        
            
        ### element-wise mutliplication
        # if c is a vector of ones, then this doesn't change anything
        c_fitcoeff = jnp.multiply(fit_coeff, c) #(B, 1)
        
        # sow outputs again, if you applied expOut
        if store_interms and self.expOut:
            self.sow_histograms_scalars(mat = c_fitcoeff,
                                        label = f"{self.name}/fit_coeff x C",
                                        which = ['hists','scalars'])
            
        # return fit_coeff, both before and after multiplying by c
        return (fit_coeff, c_fitcoeff)
    
    
    
    ###############
    ### HELPERS   #
    ###############
    def sow_histograms_scalars(self, mat, label, which=['hists','scalars']):
        """
        helper to sow intermediate values in the dense projection layers
        """
        if 'hists' in which:
            # full histograms
            self.sow("histograms",
                     label,
                     mat,
                     reduce_fn = lambda a, b: b)
        
        if 'scalars' in which:
            # scalars
            out_dict = summary_stats(mat=mat, key_prefix=label)
            for key, val in out_dict.items():
                self.sow("scalars",
                         key,
                         val,
                         reduce_fn = lambda a, b: b)



#################
### test it out #
#################
if __name__ == '__main__':
    import flax
    import optax
    from flax.training import train_state

    # additional key entry for trainstate object
    class TrainState(train_state.TrainState):
        key: jax.Array
    
    
    # fake in
    seq = jnp.array([[1,2,3,4,0,0,1],
                     [1,2,3,4,5,6,1]])
    
    
    # init
    dummy_in = jnp.empty(seq.shape)
    tx = optax.adam(learning_rate = 0.001)
    
    model = FcnnReg(dense_size_lst = [7, 4],
                     activation_fn = nn.relu,
                     expOut=False,
                      name='test_module')
    init_params = model.init(rngs=jax.random.key(0),
                             seq_c = dummy_in,
                             training = False,
                             store_interms = False)
    
    tstate = TrainState.create(apply_fn = model.apply,
                               params = init_params,
                               key = jax.random.key(0),
                               tx = tx)
    
    
    # apply on fake input to test behavior
    out, out_dict = tstate.apply_fn(variables = init_params,
                                    mutable = ['histograms','scalars'],
                                    seq_c = seq,
                                    training = True,
                                    store_interms = True)
    