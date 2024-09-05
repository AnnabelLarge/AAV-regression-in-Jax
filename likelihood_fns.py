#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:16:30 2023

@author: annabel_large


jit-compatible: True

ABOUT:
======
calculate the full log-likelihoods

to be honest, not sure why I have these, but they're easy enough to 
  convert, so keep them I guess
"""
import jax
from jax import numpy as jnp
from jax.scipy.special import gammaln
import math



##########################
### Full log-likelihoods #
##########################
def gaussian_logll(y, mu, var):
    """
    -ln{sqrt(2 math.pi var)} + {-(1/2) * var * (y-mu)^2}
    """
    ### calculate the probability
    # -ln{sqrt(2 π σ^2)}
    lead_const = -jnp.log( jnp.sqrt(2 * math.pi * var) )
    
    # -(1/2) * σ^2 * (y-μ)^2
    square_term = -0.5 * var * jnp.square(y - mu) 
    
    # -ln{sqrt(2 π σ^2)} + {-(1/2) * σ^2 * (y-μ)^2}
    ll_persample = lead_const + square_term
    
    return ll_persample


def poisson_logll(y, lam):
    """
    yln(lam) - lam - lgamma(y+1)
    """
    ### calculate the probability
    # yln( | λ | )
    term1 = y * jnp.log(lam)
    
    # ln( | Γ(y+1) | )
    term3 = gammaln(y+1)
    
    # yln( | λ | ) - λ - ln( | Γ(y+1) | )
    ll_persample = term1 - lam - term3
    
    return ll_persample


def negbinom_logll(y, r, p):
    """
    lgamma(y+r) - lgamma(r) - lgamma(y+1) + yln(1-p) + rln(p)
    """
    ### calculate the probability
    # ln( | Γ(y+r) | )
    lgamma_y_r = gammaln(y+r)
    
    # ln( | Γ(r) | )
    lgamma_r = gammaln(r)
    
    # ln( | Γ(y+1) | )
    lgamma_y_1 = gammaln(y+1)
    
    # yln(1-p)
    term4 = y*jnp.log(1-p)
    
    # rln(p)
    term5 = r*jnp.log(p)
    
    # ln( | Γ(y+r) | ) - ln( | Γ(r) | ) - ln( | Γ(y+1) | ) + yln(1-p) + rln(p)
    ll_persample = lgamma_y_r - lgamma_r - lgamma_y_1 + term4 + term5
    
    return ll_persample
