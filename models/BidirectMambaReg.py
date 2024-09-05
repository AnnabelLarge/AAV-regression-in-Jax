#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:30:33 2024

@author: annabel
"""
# general python
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field
import math
from functools import reduce
import einops

# flax n jax
import jax
import jax.numpy as jnp
import flax.linen as nn

# custom
from utils import summary_stats
from models.Mamba_based.blocks import (embedding_with_padding, 
                                       SelectiveSSM, 
                                       mlp_layer)

SOW_REDUCE = lambda a, b: b



class BidirectMambaReg(nn.Module):
    """
    build features with a bidirectional Mamba model, 
      then project to output scores
    
    inputs should be categorically encoded and un-aligned
    
    """
    ### inputs related to this AAV project
    dense_size_lst: list
    expOut: bool
    name: str
    
    ### inputs unique to this class
    expansion_factor: float  # E
    hidden_dim: int #D
    n_blocks: int = 1
    norm_type: str = "rms"
    bn_momentum: float = 0.9
    
    ### inputs shared with/fed into SelectiveSSM
    ssm_hidden_features: int = 16   # N
    ssm_shift_conv_size: int = 3
    activation: str = "silu"
    dt_rank: Union[int, str] = 'auto'
    dt_proj: bool = True   # whether to use a linear projection (vs broadcast) to map dt_rank to D
    which_metrics: tuple = field(default_factory=tuple)
    padding_idx: int = 0
    base_alphabet_size: int = 21
    
    ### inputs to keep as-is
    complement: bool = False # not relevant for protein models
    tie_in_proj: bool = False 
    tie_gate: bool = False
    
    
    @nn.compact
    def __call__(self, seq_c, training: bool, store_interms: bool):        
        ### unpack sequences
        seq = seq_c[:, :-1] #(B, featSize)
        c = seq_c[:, -1][:,None] #(B, 1)
        
        
        ######################
        ### 0: INITIAL EMBED #
        ######################
        # f(seq) = x
        x, padding_mask = embedding_with_padding(num_embeddings = self.base_alphabet_size, 
                                             features = self.hidden_dim, 
                                             padding_idx = self.padding_idx,
                                             name = f'{self.name}/initial embed')(x = seq)
        # x is (B, L, D)
        
        ######################
        ### 1: SETUP FOR SSM #
        ######################
        # get dimensions for parameter blocks
        input_features = x.shape[-1]  # D
        
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(input_features / 16)
        else:
            dt_rank = self.dt_rank

        # choose an activation function to use later
        if self.activation == "gelu":
            activate = nn.gelu
        elif self.activation == "silu":
            activate = nn.silu
        elif self.activation == "relu":
            activate = nn.relu
        else:
            raise Exception(f"Unknown activation: {self.activation}")
        
        
        ############################
        ### 2: INTERMEDIATE BLOCKS #
        ############################
        for i in range(self.n_blocks-1):
            # save input before block (B,L,D)
            skip = x
            
            # norm -> ssm
            x_forward, x_reverse = self.apply_SSM(x=x,
                                                  input_features=input_features, 
                                                  dt_rank = dt_rank,
                                                  activate = activate,
                                                  training = training,
                                                  store_interms = store_interms,
                                                  block_idx=i) # each are (B, L, E*D)
            
            # for internal blocks, concatenate and project
            x = jnp.concatenate ([x_forward, x_reverse], axis=-1) # (B, L, 2*E*D)
            
            if store_interms:
                self.sow ("scalars", 
                          f"{self.name}/Mamba block {i}.2,3- SSM GATING/gated_mean", 
                          jnp.mean(x),
                          reduce_fn = SOW_REDUCE)
                self.sow ("scalars", 
                          f"{self.name}/Mamba block {i}.2,3- SSM GATING/gated_sd", 
                          jnp.std(x),
                          reduce_fn = SOW_REDUCE)
            
            # project from (B, L, 2*E*D) -> (B,L,D)
            x = nn.Dense (features=input_features, 
                          name=f'{self.name}/Mamba block {i}.4- proj from SSM', 
                          kernel_init=nn.initializers.lecun_normal()) (x)
    
            # (output diagnostics)
            if store_interms:
                self.sow ("scalars", 
                          f"{self.name}/Mamba block {i}.4- projection from SSM/residual_mean", 
                          jnp.mean(x),
                          reduce_fn = SOW_REDUCE)
                self.sow ("scalars", 
                          f"{self.name}/Mamba block {i}.4- projection from SSM/residual_sd", 
                          jnp.std(x),
                          reduce_fn = SOW_REDUCE)
    
            # residual add
            x = skip + x
        
        
        ####################
        ### 3: FINAL BLOCK #
        ####################
        x_final_forward, x_final_reverse = self.apply_SSM(x=x, 
                                                          input_features=input_features,
                                                          dt_rank = dt_rank,
                                                          activate = activate,
                                                          training = training,
                                                          store_interms = store_interms,
                                                          block_idx = self.n_blocks-1) # each are (B, L, E*D)
        
        if store_interms:
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {self.n_blocks-1}.2,3- SSM GATING FW/gated_mean", 
                      jnp.mean(x_final_forward),
                      reduce_fn = SOW_REDUCE)
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {self.n_blocks-1}.2,3- SSM GATING FW/gated_sd", 
                      jnp.std(x_final_forward),
                      reduce_fn = SOW_REDUCE)
            
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {self.n_blocks-1}.2,3- SSM GATING RV/gated_mean", 
                      jnp.mean(x_final_reverse),
                      reduce_fn = SOW_REDUCE)
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {self.n_blocks-1}.2,3- SSM GATING RV/gated_sd", 
                      jnp.std(x_final_reverse),
                      reduce_fn = SOW_REDUCE)
        
        # take the LAST hidden state of x_forward and the FIRST hidden 
        #   state of x_reverse, and concatenate these
        x = jnp.concatenate( [x_final_forward[:, -1, :],
                              x_final_reverse[:,  0, :]],
                              axis = -1 ) #(B, 2*E*D)
        
        
        ########################################
        ### 4: PROJECTION FROM HIDDEN STATE TO #
        ###    FINAL LOG-RATIO SCORE           #
        ########################################
        ### repeat (dense -> relu) blocks
        for j, out_size in enumerate(self.dense_size_lst):
            x = nn.Dense(features=out_size, use_bias=True)(x)
            x = nn.relu(x)
            
            if store_interms:
                self.sow_histograms_scalars(mat = x,
                                            label = f'{self.name}/dense projection {j}',
                                            which = ['scalars'])
        
        # final projection to scores
        fit_coeff = nn.Dense(features=1, use_bias=True)(x) # (B, 1)
        
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
    def apply_SSM(self, x, input_features, dt_rank, activate, training,
                  store_interms, block_idx):
        """
        applies the SSM and returns the forward and reverse 
          sequence encodings
        """
        # normalize input; default is RMS norm
        # note: if you use batchnorm, I think it's still technically 
        # affected by padding characters in the batch
        # if self.norm_type == "batch":
        #     x = nn.BatchNorm(momentum=self.bn_momentum, use_running_average=not train)(x)
        if self.norm_type == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm_type == "group":
            x = nn.GroupNorm()(x)
        elif self.norm_type == "rms":
            x = nn.RMSNorm()(x)


        # project to expanded dimension (take care of both directions 
        #   in one dense layer)
        ED = math.ceil (self.expansion_factor * input_features)
        n_in_proj = 1 if self.tie_in_proj else 2
        n_gate = 1 if self.tie_gate else 2
        [xf, _xr, zf, _zr] = jnp.split (nn.Dense (features=((n_in_proj+n_gate)*ED), 
                                                  name=f'{self.name}/Mamba block {block_idx}.1- proj to SSM)', 
                                                  kernel_init=nn.initializers.lecun_normal()) (x), 
                                        [k*ED for k in [1,n_in_proj,n_in_proj+1]], 
                                        axis=-1)
        xr = xf if self.tie_in_proj else _xr
        zr = zf if self.tie_gate else _zr
        
        
        # run forward and backward SSM
        xf = SelectiveSSM(layer_prefix = f'{self.name}/Mamba block {block_idx}.2- FW SSM',
                          which_metrics = self.which_metrics,
                          reverse=False, 
                          hidden_features=self.ssm_hidden_features, 
                          dt_rank=dt_rank,
                          dt_proj=self.dt_proj,
                          shift_conv_size=self.ssm_shift_conv_size,
                          activation=self.activation,
                          padding_idx=self.padding_idx,
                          name=f'{self.name}/Mamba block {block_idx}.2: FW SSM') (xf, training)
        
        xr = SelectiveSSM(layer_prefix = f'{self.name}/Mamba block {block_idx}.3- RV SSM',
                          which_metrics = self.which_metrics,
                          reverse=True,
                          complement=self.complement, 
                          hidden_features=self.ssm_hidden_features, 
                          dt_rank=dt_rank,
                          dt_proj=self.dt_proj,
                          shift_conv_size=self.ssm_shift_conv_size,
                          activation=self.activation,
                          padding_idx=self.padding_idx,
                          name=f'{self.name}/Mamba block {block_idx}.3- RV SSM')(xr, training)

        # (output diagnostics)
        if store_interms:
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {block_idx}.2- FW SSM/gate_fwd_mean", 
                      jnp.mean(zf),
                      reduce_fn = SOW_REDUCE)
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {block_idx}.2- FW SSM/gate_fwd_sd", 
                      jnp.std(zf),
                      reduce_fn = SOW_REDUCE)
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {block_idx}.3- RV SSM/gate_rev_mean", 
                      jnp.mean(zr),
                      reduce_fn = SOW_REDUCE)
            self.sow ("scalars", 
                      f"{self.name}/Mamba block {block_idx}.3- RV SSM/gate_rev_sd", 
                      jnp.std(zr),
                      reduce_fn = SOW_REDUCE)
        
        ### multply by gates and return values
        x_forward = xf * activate(zf) #(B, L, 2D)
        x_reverse = xr * activate(zr) #(B, L, 2D)
        
        return (x_forward, x_reverse)
    
    
    
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
    seq = jnp.array([[1,2,3,4,0,0,1]])
    
    
    # init
    dummy_in = jnp.empty(seq.shape, dtype=int)
    tx = optax.adam(learning_rate = 0.001)
    
    model = BidirectMambaReg(expansion_factor=2,
                             hidden_dim=8,
                             dense_size_lst=[4],
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
