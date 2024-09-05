#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:24:25 2023

@author: annabel_large


jit-compatible: True

ABOUT:
======
All __call__ have trace of: (model_out, n_pre_true, n_post_true)

Gaussian-based:
    - OLS (or rather, a wrapper for the built-in function)
    - WLS

Poisson-based
    - Poisson Deviance

Negative binomial-based:
    - NB-C Deviance (constant dispersion)
    - NB-SS Deviance (sample-specific dispersion)

"""
import jax
from jax import numpy as jnp
import flax.linen as nn
import optax
from jax.typing import ArrayLike

from typing import Union

from likelihood_fns import *


###############################################################################
### GAUSSIAN-BASED ############################################################
###############################################################################
class MSELoss_wrapper(nn.Module):
    """
    model_out = log(lambda) = XB
    """
    
    N_pre: int
    N_post: int
    
    @nn.compact
    def __call__(self, model_out, n_pre_true, n_post_true):
        """
        a wrapper for the built-in optax.squared_error()
        also calculates the true ratios
        """
        # calculate the true log ratio
        true_ratios = self.calc_log_ratio(n_pre_true, n_post_true)

        # calculate the loss for each sample
        sq_err = optax.squared_error(predictions=model_out, 
                                     targets=true_ratios)
        
        # weight by sample variance
        var = self.calc_var(n_pre_true, n_post_true)
        loss_per_samp = jnp.multiply(sq_err, var)
        return loss_per_samp
    
    def retrieve_expOut_addC(self):
        expOut = False
        addC = False
        return (expOut, addC)
    
    def retrieve_alpha(self):
        # this is only for models where addC==True, but 
        # place here for consistency I guess
        return 0
    
    def calc_log_ratio(self, n_pre, n_post):
        num = n_post/self.N_post
        denom = n_pre/self.N_pre
        ratio = (num/denom)
        log_ratio = jnp.log(ratio)
        return log_ratio
    
    def calc_var(self, n_pre, n_post):
        """
        in mean squared error, all have variance of one
        """
        return jnp.ones( n_pre.shape )
    
    def log_llhood(self, model_out, n_pre_true, n_post_true):
        """
        the full gaussian log likelihood
          -ln{sqrt(2 π σ^2)} + {-(1/2) * σ^2 * (y-μ)^2}
        """
        # calculate the true log ratio
        true_ratios = self.calc_log_ratio(n_pre_true, n_post_true)
        
        # calculate the variance from the true counts
        var = self.calc_var(n_pre_true, n_post_true)
        
        # calculate the joint log probability
        ll_persample = gaussian_logll(y = true_ratios, 
                                      mu = model_out, 
                                      var = var)
        return ll_persample


class WLSLoss(MSELoss_wrapper):
    """
    The WLS loss, as outlined in Zhu et al
    model_out = log(lambda) = XB
    
    essentially the same as the MSELoss_wrapper, but now samples
      are weighted with new variance term
    """
    
    N_pre: int
    N_post: int
    
    def calc_var(self, n_pre, n_post):
        ### (1/n_post)(1 - (n_post/N_post)) + (1/n_pre)(1 - (n_pre/N_pre))
        term1 = (1/n_post)*(1-(n_post/self.N_post))
        term2 = (1/n_pre)*(1-(n_pre/self.N_pre))
        var = term1 + term2
        return var
    


###############################################################################
### POISSON-BASED #############################################################
###############################################################################
class PoissonDev(nn.Module):
    """
    Implement Deviance loss, based on Poisson distribution
    loss = -ave{yln(λ) - λ}
    y_pred = λ = exp{w^T * x}
    """
    @nn.compact
    def __call__(self, model_out, n_pre_true, n_post_true):
        # get the likelihood of the saturated model
        loglike_sat = self.loglike_helper(y = n_post_true, lam = n_post_true)
        
        # get the likelihood of the candidate model
        loglike_pred = self.loglike_helper(y = n_post_true, lam = model_out)
        
        # find the deviance loss per sample
        loss_per_samp = 2*(loglike_sat - loglike_pred)
        return loss_per_samp
        
    def retrieve_expOut_addC(self):
        expOut = True
        addC = True
        return (expOut, addC)
    
    def retrieve_alpha(self):
        # could probably introduce skew here, but for now, just
        # return dummy variable 0
        return 0
    
    def loglike_helper(self, y, lam):
        """
        yln(λ) - λ
        
        used explicitly in calculating deviance loss
        """
        return y*jnp.log(lam) - lam
    
    def log_llhood(self, model_out, n_pre_true, n_post_true):
        """
        this calculates the full log-likelihood:
        yln(λ) - λ - ln(Γ(y+1))
        """
        ll_persample = poisson_logll(y = n_post_true, 
                                  lam = model_out)
        return ll_persample
    


###############################################################################
### NEGATIVE BINOMIAL BASED ###################################################
###############################################################################
class NegBinomDev(nn.Module):
    """
    Implement Deviance loss with NB2 and constant dispersion param (provided)
    y_pred = exp{w^T * x}
    r = dispersion_param * y_pred
    p = dispersion_param / (1+dispersion_param)
    
    TODO:
    -----
    For now, phi can either be a float (NB-Const) or calculated externally 
      per sampled (NB-SS)
    
    May want to change to a dedicated method, like WLS in the future?
    """
    
    phi: Union[float, ArrayLike]
    alpha: float
    @nn.compact
    def __call__(self, model_out, n_pre_true, n_post_true):
        # get the likelihood of the saturated model
        loglike_sat = self.LogLike(y = n_post_true, lam = n_post_true)
        
        # get the likelihood of the candidate model
        loglike_pred = self.LogLike(y = n_post_true, lam = model_out)
        
        # find the deviance loss per sample
        loss_per_samp = 2*(loglike_sat - loglike_pred)
        return loss_per_samp
        
    def retrieve_expOut_addC(self):
        expOut = True
        addC = True
        return (expOut, addC)
    
    def retrieve_alpha(self):
        """
        todo: why is this here? just call instance.alpha?
        """
        return self.alpha
    
    def loglike_helper(self, y, lam):
        """
        yln(λ) - (y+φ)ln(λ+φ)
        """
        return y*jnp.log(lam) - (y+self.phi)*jnp.log(lam+self.phi)
    
    def log_llhood(self, model_out, n_pre_true, n_post_true):
        """
        this calculates the full log-likelihood:
        ln(Γ(y+r)}) - ln(Γ(r)) - ln(Γ(y+1)) + yln(1-p) + rln(p)
        """
        # calculate logll
        ll_persample = negbinom_logll(y = n_post_true, 
                                      r = self.phi, 
                                      p = (self.phi) / (self.phi + model_out))
        return ll_persample

