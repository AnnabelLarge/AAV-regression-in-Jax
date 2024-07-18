#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 18:48:28 2022

@author: annabel_large

"""
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp
from jax.tree_util import tree_map

import pandas as pd
import numpy as np

from utils import (OneHotTokenPad, CharTokenizer)

def jax_collator(batch):
    return tree_map(jnp.asarray, default_collate(batch))

class ChimericLoader(Dataset):
    """
    initialize with:
    1. dataset file prefix
    2. addc (bool)
    3. maxlen
    4. alpha
    5. transform
    6. addone: add a pseudocount of 1 (default is true)
    
    returns (in this order):
    1. dataframe index (int)
    2. n_pre (int)
    3. n_post_R1 (int)
    4. features: sequences + c (torch tensor)
    
    (Ignore n_post_R2 for now)
    """
    def __init__(self, prefix, addc, maxlen, alpha, transform, addone=True):
        ########################################
        ### make sure valid options are chosen #
        ########################################
        # prefix
        valid_prefixes = ['train','valid']
        assert prefix in valid_prefixes, f'Options are: {valid_prefixes}'
        
        # addc
        assert type(addc) == bool, "AddC option needs to be bool"
        
        # addone
        assert type(addone) == bool, 'AddOne option needs to be bool'
        
        # transform
        valid_transforms = ['onehot_gapped','char_tokenizer', 'esm1v_mean', 
                            None]
        assert transform in valid_transforms, f'Options are: {valid_transforms}'
        
        self.transform = transform
        self.maxlen = maxlen
        
        ############################
        ### hardcode N_pre, N_post #
        ############################
        if (prefix in ['train','valid']):
            if addone:
                N_pre = 1895076
                N_post = 3084188
                
            elif not addone:
                N_pre = 1622204
                N_post = 2811316
                
        ########################
        ### load the dataframe #
        ########################
        filename = f'./data/{prefix}_seqs.tsv'
        print(f'Reading data from {filename}\n')
        self.df = pd.read_csv(filename, sep='\t')
        del filename
        
        #######################
        ### ADDONE ############
        #######################
        if addone:
            print('Adding pseudocount of one to all counts\n')
            self.df['Prepackaging_Counts'] += 1
            self.df['Postpackaging-R1_Counts'] += 1
           
        #######################
        ### TRANSFORM #########
        #######################
        # One-hot encoding of gapped protein sequences
        # overwrite maxlen to be the fixed-width gapped protein sequence
        if transform == 'onehot_gapped':
            # toss some of the samples, due to incorrect gapped seqs
            self.df = self.df[self.df['keep_gappedSeq']]

            print('Applying OneHot encoding of gapped sequences\n')
            alphabet = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            
            self.xdim = (self.maxlen * len(alphabet)) + 1
            self.mapping = dict((c, i) for i, c in enumerate(alphabet))
        
        # categorical tokenization
        elif transform == 'char_tokenizer':
            print(('Applying character-level tokenization with padding, '+
                   'start, and end tokens\n'))
            alphabet = ['<pad>', '<bos>', '<eos>', 'A', 'C', 'D', 'E', 'F', 
                        'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 
                        'T', 'V', 'W', 'Y']
            self.xdim = (self.maxlen + 2) + 1
            self.mapping = dict((c, i) for i, c in enumerate(alphabet))
        
        # Load ESM-1v mean embeddings
        elif transform == 'esm1v_mean':
            print('Reading from pre-computed ESM-1v embeddings')
            filename = f'./data/{prefix}_ESM1v_embeddings.pt'
            self.tens = torch.load(filename)
            self.xdim = self.tens.shape[-1] + 1
        
        
        elif transform == None:
            self.xdim = (self.maxlen) + 1
            
        ###################
        ### ADD C #########
        ###################
        if addc:
            print(('Appending extra feature: '+
                  f'c = (n_pre + {alpha} * N_pre) (N_post/N_pre) \n'))
            shifted_cnt = self.df['Prepackaging_Counts'] + (alpha * N_pre)
            self.df['c'] = shifted_cnt * (N_post/N_pre)
        
        elif not addc:
            print('Adding 1 to end of feature vec (placeholder for c-vec)\n')
            self.df['c'] = 1
    
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        
        # ### Get ChimericID (string)
        # sample_readname = sample['ChimericID']
        
        ### Get n_pre and n_post (int)
        sample_pre = sample['Prepackaging_Counts'] 
        sample_pre = torch.Tensor([sample_pre])
        sample_post = sample['Postpackaging-R1_Counts'] 
        sample_post = torch.Tensor([sample_post])
        
        
        ###################################
        ### Get Protein Sequence (tensor) #
        ###################################
        ### one-hot encoding of gapped (aligned) sequences
        if self.transform == 'onehot_gapped':
            seq = sample['Gapped_Prot_Seq']
            sample_feats = OneHotTokenPad(seq = seq, 
                                          max_len = self.maxlen, 
                                          mapping = self.mapping)
        
        ### categorical encoding with end padding
        elif self.transform == 'char_tokenizer':
            seq = sample['Prot_Seq']
            sample_feats = CharTokenizer(seq = seq, 
                                         max_len = self.maxlen, 
                                         mapping = self.mapping)
        
        ### loading from ESM-1v embeddings
        elif self.transform == 'esm1v_mean':
            sample_feats = self.tens[idx]
        
        
        ##############################
        ### Add c (or placeholder 1) #
        ##############################
        # if you're not adding c, then this will just be 1
        final_feat = torch.Tensor([ sample['c'] ])
        sample_feats = torch.cat([sample_feats, final_feat], dim=0)
        
        return (idx, sample_pre, sample_post, sample_feats)
    
    
    def get_xdim(self):
        """
        Other code used this function, so keep for now...
        """
        return self.xdim
    
        


#########################
### TEST FUNCTIONS HERE #
#########################
if __name__ == '__main__':
    y = ChimericLoader(prefix='valid', 
                        addc = False,
                        alpha=0,
                        maxlen = 20,
                        transform='onehot_gapped')
    ex_y = y[0]
    
    y_dl = DataLoader(y, 
                      batch_size = 4, 
                      shuffle = True,
                      collate_fn = jax_collator)
    
    ex_batch = list(y_dl)[0]
    idxes, n_pres, n_posts, feats = ex_batch
