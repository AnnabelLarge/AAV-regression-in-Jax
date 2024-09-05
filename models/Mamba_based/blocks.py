#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:13:13 2024

@author: annabel

ABOUT:
======
Blocks and pieces to use in Mamba state-space models (mostly from 
  Ian's selectssm.py)

using what he says is the fastest implementation, ssm_chunked_scan

"""
import logging
from typing import Any, Callable, Sequence, Union, Tuple
from dataclasses import field
import math
from functools import reduce

import jax
import jax.numpy as jnp
import einops
import flax.linen as nn


######################
### helper functions #
######################
def l2_norm(params, alpha = 1.):
    """
    don't actually need this, because I'll use weight decay in Adam
    """
    return alpha * jnp.sum (jnp.array ([jnp.sum(x*x) for x in jax.tree_util.tree_leaves(params)]))

def inverse_softplus(x):
    return x + jnp.log(1 - jnp.exp(-x))

def debug_log(fmt: str, *args, **kwargs):
    jax.debug.callback(
        lambda *args, **kwargs: logging.warning(fmt.format(*args, **kwargs)),
        *args, **kwargs)

def largest_factor_up_to(b,n):
    if n < 2:
        return n
    k = b
    while n % k != 0:
        k -= 1
    return k


###############################
### embedding/encoding layers #
###############################
class embedding_with_padding(nn.Module):
    """
    replicated torch's embedding function, with padding_idx option
    """
    num_embeddings: int
    features: int
    name: str
    padding_idx: int=0
    
    @nn.compact
    def __call__(self, x):
        embedded = nn.Embed(num_embeddings = self.num_embeddings, 
                            features = self.features)(x)
        
        # mask positions with padding tokens
        masking = jnp.where(x == self.padding_idx, 0, 1)
        masking = jnp.expand_dims(masking, axis = 2)
        masking = jnp.repeat(masking, repeats = self.features, axis=2)
        
        out = jnp.multiply(embedded, masking)
        return (out, masking)


################
### simple MLP #
################
class mlp_layer(nn.Module):
    input_features: int
    mlp_expansion: int
    name: str
    activation: str='silu'
    mlp_dropout: float = 0.1
    padding_idx: int = 0
    
    @nn.compact
    def __call__(self, x, training: bool, sow_activations: bool):
        SOW_REDUCE = lambda a, b: b
        
        x = nn.Dense(self.mlp_expansion * self.input_features, 
                     name="mlp_expand", 
                     kernel_init=nn.initializers.lecun_normal())(x)
        
        x = nn.Dropout(rate=self.mlp_dropout, 
                       deterministic=not training)(x)
        
        if self.activation == "gelu":
            x = nn.gelu(x)
        elif self.activation == "relu":
            x = nn.relu(x)
        elif self.activation == "silu":
            x = nn.silu(x)
        elif self.activation is not None:
            raise Exception(f"Unknown activation: {self.activation}")
        
        if sow_activations:
            self.sow ( "metrics",
                       f"{self.name}, after mlp_expand and ReLU",
                       datamat,
                       reduce_fn = SOW_REDUCE)
        
        x = nn.Dense(self.input_features, 
                     name="mlp_proj", 
                     kernel_init=nn.initializers.lecun_normal())(x)
        
        x = nn.Dropout(rate=self.mlp_dropout, 
                       deterministic=not training)(x)
        
        return x
        


#######################################
### scan implementation for SSM layer #
#######################################
# x: (B, L, D)
# Acoeff: (D, N)
# Bcoeff: (B, L, N)
# Ccoeff: (B, L, N)
# dt: (B, L, D) or (B, L, 1);  can assume (B, L, D) and rely on broadcasting
def ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, chunk_size: int = None, n_channel_groups: int = 1):
    B = x.shape[-3]
    L = x.shape[-2]
    D = x.shape[-1]
    N = Acoeff.shape[-1]

    if n_channel_groups is not None:
        K = n_channel_groups
    else:
        K = 1
    if D % K != 0:
        raise ValueError(f"n_channel_groups={n_channel_groups} must divide D={D}")

    if chunk_size is None:
        chunk_size = largest_factor_up_to(int(math.sqrt(K*L)),L)

    if L % chunk_size != 0:
        raise ValueError(f"chunk_size={chunk_size} must divide L={L}")
    n_chunks = L // chunk_size

    # Transpose length & batch dimensions to make the scan over length, and 
    # split into chunks
    # This is a bit inefficient, but taking dynamic slices appears to be worse
    x_chunks = einops.rearrange (x, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)
    A_blocks = einops.rearrange (Acoeff, '(k d) n -> k d n', k=K)
    B_chunks = einops.rearrange (Bcoeff, 'b (c l) n -> c l b n', c=n_chunks)
    C_chunks = einops.rearrange (Ccoeff, 'b (c l) n -> c l b n', c=n_chunks)
    dt_chunks = einops.rearrange (dt, 'b (c l) (k d) -> c k l b d', c=n_chunks, k=K)

    # Function to do an associative scan for a single chunk
    # We decorate this with @jax.remat to flag that we are OK with re-performing this scan whenever needed
    ### todo: really make sure padding characters don't affect anything here
    @jax.remat
    def scan_chunk (carry, chunk):
        # For the purposes of shape annotation within this code we write D instead of D/K
        g_init, h_init = carry  # (1, B, D, N)  (1, B, D, N)

        x_chunk, A_block, B_chunk, C_chunk, dt_chunk = chunk
        # dA = exp(A*dt) [zero-order hold], dB = B*dt*x [Euler step]
        dA = jnp.exp (jnp.einsum ('dn,lbd->lbdn', A_block, dt_chunk))  # (chunk_size, B, D, N)
        dB = jnp.einsum ('lbn,lbd,lbd->lbdn', B_chunk, x_chunk, dt_chunk)  # (chunk_size, B, D, N)
        # The associative scan is a product of matrices of the form 
        # ((g,h),(0,1)) where g_i=exp(A*dt)x_i and h_i=B*dt*x_i
        # Since matrices of this form are are closed under multiplication, 
        # we can represent all intermediate products in the same way
        @jax.remat
        def associative_scan_fn (l, r):  # l, r, and return value are tuples of the form ((B,D,N), (B,D,N))
            g_l, h_l = l
            g_r, h_r = r
            return tuple((g_l*g_r, g_r*h_l + h_r))
        gs, hs = jax.lax.associative_scan (associative_scan_fn, (dA, dB))  # (chunk_size, B, D, N)  (chunk_size, B, D, N)
        hs = gs * h_init + hs  # Incorporate h_init here so that it is reflected in y_chunk
        # We only need to keep the last state of gs, so we can discard the 
        # rest. Otherwise we would incorporate g_init here, like so:
        # gs = g_init * gs
        y_chunk = jnp.einsum ('lbn,lbdn->lbd', C_chunk, hs)  # (chunk_size, B, D)
        return (gs[-1:,...] * g_init, hs[-1:,...]), y_chunk  # note g_init incorporated here

    # A wrapper that splits the dimensions into K blocks and does the inner 
    # associative scan for each block, re-using B and C (which don't 
    # change across dimensions)
    @jax.remat
    def scan_chunk_mapped (carry, chunk):
        g_init, h_init = carry  # (K,1,B,D/K,N) (K,1,B,D/K,N)
        
        x_chunk, B_chunk, C_chunk, dt_chunk = chunk   # (K,B,L,D/K), (B,L,N), (B,L,N), (K,B,L,D/K)
        @jax.remat
        def scan_chunk_wrapper (block):
            dA_init_block, dB_init_block, x_chunk_block, A_block, dt_chunk_block = block
            return scan_chunk ((dA_init_block, dB_init_block), (x_chunk_block, A_block, B_chunk, C_chunk, dt_chunk_block))
        return jax.lax.map (scan_chunk_wrapper, (g_init, h_init, x_chunk, A_blocks, dt_chunk))

    # Perform the scan over chunks recurrently (with rematerialization as 
    # noted above), with each chunk being an associative scan
    (_A_final, _h_final), y_chunks = jax.lax.scan (scan_chunk_mapped, 
                                                   (jnp.ones((K,1,B,D//K,N)), 
                                                    jnp.zeros((K,1,B,D//K,N))), 
                                                   (x_chunks, B_chunks, C_chunks, dt_chunks) )  # (K, n_chunks, B, D//K)

    return einops.rearrange (y_chunks, 'c k l b d -> b (c l) (k d)')  # (B, L, D)



######################
### SSM Layer + Conv #
######################
class SelectiveSSM(nn.Module):
    """ 
    A variation on MAMBA: https://arxiv.org/pdf/2312.00752.pdf 
    
    init with:
    ----------
    things I could play with
        - layer_prefix:
                this string is the prefix prepended to the metrics dictionary entry 
        
        - which_metrics [default=empty list]:
                this list contains which metrics to record from this intermediate layer
            
        - reverse [default=False]: 
                used in bidirectional mamba
                
        - hidden features, N [default=16]: 
                the lower-dimensional embedding for SSM layer
                
        - dt_rank [default="auto"]: 
                size of dt variable; if I change this later, assert D % dt_rank == 0
                
        - dt_proj [default=True]:
                whether or not to automatically learn dt initialization
                
        - shift_conv_size [default=3]: 
                the kernel size for the initial 1D convolution
                
        - activation [default: 'silu']:
                activation function i.e. the gating mechanism for the SSM layer
            
    things I should not change
        - complement (I think this is only relevant for DNA models)
        - chunk_size (let recursive scan automatically determine this)
        - n_channel_groups (let recursive scan automatically determine this)
        - dt_min, dt_max (these were originally defined in mamba paper and 
                are probably find to keep)
        
        
    apply_fn:
    ---------
    inputs for apply_fn
        - x: matrix of size (B, L, D)
        - training: if model is in training mode or not (only matters for 
                    recording diagnostics)
    
    outputs from apply_fn
        - y: matrix of size (B, L, D)
    """
    ### inputs to vary in hyperparam sweeps
    layer_prefix: str
    which_metrics:  tuple = field(default_factory=tuple)
    reverse: bool = False
    
    hidden_features: int = 16  # N
    dt_rank: Union[int, str] = 'auto'  # R
    dt_proj: bool = True   # whether to use a linear projection (vs broadcast) to map dt_rank to D
    shift_conv_size: int = 3
    activation: str = "silu"
    padding_idx: int = 0
    
    
    ### inputs to keep as-is
    complement: bool = False  # only checked if reverse is true
    chunk_size: int = None
    n_channel_groups: int = None
    dt_min: float = 0.001  # 1/(long-range context length)
    dt_max: float = 0.1    # 1/(short-range context length)


    @nn.compact
    def __call__(self,
                 x,  # (B, L, D)
                 training: bool = False):
        ### for metrics: default behavior is to only keep result from most
        ### recent occurence
        SOW_REDUCE = lambda a, b: b
        
        ### 1: get dimensions for parameter blocks
        B = x.shape[-3]
        L = x.shape[-2]
        D = x.shape[-1]  # if called by BidirectionalMamba, this is actually E*D
        # E: expansion factor

        N = self.hidden_features
 
        if self.dt_rank == 'auto':
            dt_rank = math.ceil(D / 16)
        else:
            dt_rank = self.dt_rank

        # (output diagnostics)
        if training and 'seq_encoders' in self.which_metrics:
            self.sow("metrics", 
                     f"{self.layer_prefix}/ssm_input_mean", 
                     jnp.mean(x), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics",
                     f"{self.layer_prefix}/ssm_input_sd", 
                     jnp.std(x), 
                     reduce_fn = SOW_REDUCE)


        ### 2: flip input along length dimension (in bidirectional)
        if self.reverse:
            x = jnp.flip (x, axis=(-2,-1) if self.complement else -2)


        ### 3: first 1D convolution
        u = nn.Conv (features=D, 
                     feature_group_count=D, 
                     kernel_size=(self.shift_conv_size,), 
                     strides=(1,), 
                     padding="SAME", 
                     use_bias=False, 
                     name="shift_conv", 
                     kernel_init=nn.initializers.lecun_normal()) (x)  # (B, L, D)

        # (output diagnostics)
        if training and 'seq_encoders' in self.which_metrics:
            self.sow("metrics", 
                     f"{self.layer_prefix}/conv_mean", 
                     jnp.mean(u), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/conv_sd", 
                     jnp.std(u), 
                     reduce_fn = SOW_REDUCE)


        ### 4: first activation after convolution
        if self.activation == "gelu":
            u = nn.gelu(u)
        elif self.activation == "relu":
            u = nn.relu(u)
        elif self.activation == "silu":
            u = nn.silu(u)
        elif self.activation is not None:
            raise Exception(f"Unknown activation: {self.activation}")


        ### 5: SSM parameter initialization
        # 5.1: Initialize A nonrandomly with evenly spaced eigenvalues; 
        # keep parameterization in log space to guarantee A<0 (ask Ian about this)
        Acoeff = -jnp.exp (self.param ('A_log', lambda rng: jnp.log (jnp.repeat (jnp.arange(start=1,stop=N+1,dtype=jnp.float32)[None,:], 
                                                                                 D, 
                                                                                 axis=0)) ) )  # (D, N)
        
        # 5.2: initialize B and C directly from convolution output u ( i.e. x(t=0) )
        Bcoeff, Ccoeff = jnp.split (nn.Dense (features=2*N, 
                                              name='BC', 
                                              use_bias=True, 
                                              kernel_init=nn.initializers.lecun_normal()) (u), 
                                    2, 
                                    axis=-1)  # (B, L, N) *2
        
        # 5.3: initialize D, the skip connection
        Dcoeff = self.param ('D', lambda rng: jnp.ones((D,)))  # (D,)

        # 5.4: initialize delta_t, the time step, from x_0
        dt_bias_init = lambda rng, shape, dtype: inverse_softplus (jax.random.uniform (rng, 
                                                                                       shape=shape, 
                                                                                       dtype=dtype, 
                                                                                       minval=self.dt_min, 
                                                                                       maxval=self.dt_max) )
        dt = nn.Dense (features=dt_rank, 
                       use_bias=True, 
                       name='dt',
                       kernel_init=nn.initializers.lecun_normal(),
                       bias_init=nn.initializers.zeros if self.dt_proj else dt_bias_init) (u)  # (B, L, dt_rank)
        
        # (output diagnostics)
        if training and 'seq_encoders' in self.which_metrics:
            self.sow("metrics", 
                     f"{self.layer_prefix}/dt_lowrank_mean", 
                     jnp.mean(dt), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/dt_lowrank_sd", 
                     jnp.std(dt), 
                     reduce_fn = SOW_REDUCE)

        # after linear layer, get final dt; could have this be learnable, if desired
        if self.dt_proj:
            dt = nn.Dense (features=D, 
                           use_bias=True, 
                           kernel_init=nn.initializers.lecun_normal(), 
                           bias_init=dt_bias_init, 
                           name='dt_proj') (dt)  # (B, L, D)
        else:
            if dt_rank > 1:  # if dt_rank is 1, we can just rely on broadcasting, and save memory
                if D % dt_rank != 0:
                    raise ValueError(f"dt_rank={dt_rank} must divide D={D}")
                dt = jnp.repeat (dt, D // dt_rank, axis=-1)  # (B, L, D)
        dt = nn.activation.softplus (dt)  # (B, L, D) or (B, L, 1)

        # (output diagnostics)
        if training and 'seq_encoders' in self.which_metrics:
            self.sow("metrics", 
                     f"{self.layer_prefix}/activated_conv_mean", 
                     jnp.mean(u), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/activated_conv_sd", 
                     jnp.std(u), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/dt_mean", 
                     jnp.mean(dt), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/dt_sd", 
                     jnp.std(dt), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/A_mean", 
                     jnp.mean(Acoeff), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/A_sd", 
                     jnp.std(Acoeff), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/B_sd", 
                     jnp.std(Bcoeff), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/C_sd", 
                     jnp.std(Ccoeff), 
                     reduce_fn = SOW_REDUCE)


        ### 6: Perform SSM scan, using scan function above
        y = ssm_chunked_scan (x, Acoeff, Bcoeff, Ccoeff, dt, 
                              chunk_size=self.chunk_size, 
                              n_channel_groups=self.n_channel_groups)  # (B, L, D)


        ### 7: if you originally flipped the input, then flip it back to 
        ###    original orientation (along seqlen L)
        if self.reverse:
            y = jnp.flip (y, axis=(-2,-1) if self.complement else -2)

        # (output diagnostics)
        if training and 'seq_encoders' in self.which_metrics:
            self.sow("metrics", 
                     f"{self.layer_prefix}/ssm_residual_mean", 
                     jnp.mean(y), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/ssm_residual_sd", 
                     jnp.std(y), 
                     reduce_fn = SOW_REDUCE)


        ### 8: Add in the skip connection term, D
        y = y + jnp.einsum ('bld,d->bld', x, Dcoeff)

        # (output diagnostics)
        if training and 'seq_encoders' in self.which_metrics:
            self.sow("metrics", 
                     f"{self.layer_prefix}/ssm_output_mean", 
                     jnp.mean(y), 
                     reduce_fn = SOW_REDUCE)
            self.sow("metrics", 
                     f"{self.layer_prefix}/ssm_output_sd", 
                     jnp.std(y), 
                     reduce_fn = SOW_REDUCE)

        return y

        
