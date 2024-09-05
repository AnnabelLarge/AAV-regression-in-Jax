#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:35:27 2023

@author: annabel_large

ln(score) = XB
score = exp(XB)
"""
from jax.nn import one_hot
import numpy as np
from scipy.stats import pearsonr, spearmanr

####################################
### FOR DATALOADERS   ##############
####################################
def OneHotTokenPad(seq_lst, alphabet):
    """
    one-hot encode a given protein sequence; inputs will be-
        seq_lst: list of strings; the protein sequences
        alphabet: list of ascii chars, in encoding order
    
    output is a numpy array of shape (total_seqs, (max_len)*21)
    
    special tokens are:
        - = 0  (gap chars)
    Then, amino acids follow in single letter alphabetical order.
    
    """
    # ascii to categorical
    categorical_seq = []
    for seq in seq_lst:
        s = [alphabet.index(letter) for letter in seq]
        categorical_seq.append(s)
    categorical_seq = np.array(categorical_seq, dtype=int)
    
    # categorical to one-hot with jax one_hot
    onehot_seq = one_hot(categorical_seq, len(alphabet), dtype=int)
    
    # flatten
    onehot_seq = onehot_seq.reshape( (onehot_seq.shape[0],
                                      onehot_seq.shape[1] * onehot_seq.shape[2]) )
    
    return np.array(onehot_seq)


def CharTokenizer(seq_lst, max_len, alphabet):
    """
    per-Charcter tokenization of protein sequence; inputs will be-
        seq_lst: list of strings; the protein sequences
        max_len: maximum length of sequence; informs padding length
        alphabet: list of ascii chars, in encoding order
    
    output is a numpy array of shape (total_seqs, 21)
    
    special tokens are:
        <pad>=0
    Then, amino acids follow in single letter alphabetical order.
    
    """
    categorical_seq = []
    for seq in seq_lst:
        # if seq is LONGER than the max_len allowed, cut it off
        if len(seq) > max_len:
            seq = seq[0:max_len]
    
        # calculate the padding needed
        pad_len = max_len - len(seq)
        assert pad_len >= 0, f'Pad length was calculated to be {pad_len}'
    
        # string to categorical encoding
        main_seq = [alphabet.index(letter) for letter in seq]
        padding_end = [0]*pad_len
        categorical_seq.append( main_seq + padding_end )
    
    categorical_seq = np.array(categorical_seq, dtype=int)
    return categorical_seq



####################################
### FOR LOGGING   ##################
####################################
def summary_stats(mat, key_prefix):
    perc_zeros = (mat==0).sum() / mat.size
    mean_without_zeros = mat.sum() / (mat!=0).sum()
    
    out_dict = {f'{key_prefix}/MAX': mat.max().item(),
                f'{key_prefix}/MIN': mat.min().item(),
                f'{key_prefix}/MEAN': mat.mean().item(),
                f'{key_prefix}/MEAN-WITHOUT-ZEROS': mean_without_zeros.item(),
                f'{key_prefix}/PERC-ZEROS': perc_zeros.item()}
    
    return out_dict
        



# rewrite this for jax whenever you get to it
# ###################################
# ### FOR CORRELATIONS   ############
# ###################################
# class CalcScoresCorr(nn.Module):
#     """
#     Calcualte the correlation (either pearson or spearman) of the
#       predictions vs true scores
    
#     Initialize with:
#         1: N_pre: prepackaging population size
#         2: N_post: postpackaging population size
#         3: expOut: depends on the loss function used
#              > MSE, WLS: expOut=False (output is log score)
#              > Poisson, NegBinom: expOut=True (output is just score)
    
#     Calculates only LOG score correlations, regardless of whether or not
#     the model uses log-link. This makes it easier to compare to prior
#     literature
    
#     Returns:
#         1: the list of true scores
#         2: the correlation
#     """
#     def __init__(self, N_pre, N_post, expOut, corr_type=pearsonr):
#         super(CalcScoresCorr, self).__init__() 
#         self.N_pre = N_pre
#         self.N_post = N_post
#         self.expOut = expOut
#         self.corr_type = corr_type
    
#     def forward(self, model_out, n_pre_true, n_post_true):
#         # things need to match dimensions or you're gonna have a bad time
#         err_message = (f'model_out is {model_out.size()}\n' +
#                        f'n_pre_true is {n_pre_true.size()}\n' +
#                        f'n_post_true is {n_post_true.size()}')
#         assert (model_out.size()==n_pre_true.size()), err_message
#         assert (model_out.size()==n_post_true.size()), err_message
#         assert (n_post_true.size()==n_pre_true.size()), err_message
        
#         # calculate the true log(score)
#         num = n_post_true/self.N_post
#         denom = n_pre_true/self.N_pre
#         true_ratio = num/denom
#         true_log_ratio = torch.log(true_ratio)
        
#         # if using a log-link model, take log of model outputs
#         #   before calculating correlation
#         if self.expOut:
#             err_message = 'Using a log-link model, but got negative model outputs'
#             assert (model_out >= 0).all().item(), err_message
#             log_model_out = torch.log(model_out)
#         elif not self.expOut:
#             log_model_out = model_out
        
#         # move to cpu, if not already there
#         log_model_out_cpu = log_model_out.to('cpu')
#         true_log_ratio_cpu = true_log_ratio.to('cpu')

#         # detach tensors and convert to numpy arrays
#         # also need to get rid of extra dimension
#         log_model_out_np = log_model_out_cpu.squeeze(1).detach().numpy()
#         true_log_ratio_np = true_log_ratio_cpu.squeeze(1).detach().numpy()
        
#         # calculate the correlation
#         # if input is constant, or if input batch size is one, this function 
#         #   will fail
#         # return nan and filter it out
#         if ((log_model_out_np == 0).all()) or (len(log_model_out_np)==1):
#             corr = np.nan
#         else:
#             corr = self.corr_type(true_log_ratio_np, log_model_out_np)[0]
#         return (true_log_ratio_np, corr)





#########################
### TEST FUNCTIONS HERE #
#########################
if __name__ == '__main__':
    def test_encoders():
        true_oh_out = np.array( [1, 0, 0, 0, 0,
                              0, 1, 0, 0, 0,
                              0, 0, 1, 0, 0,
                              0, 0, 0, 1, 0,
                              0, 0, 0, 0, 1] )[None,:]
        oh_out = OneHotTokenPad(seq_lst = ['-ACGT'], 
                             alphabet = list('-ACGT'))
        
        assert np.allclose(true_oh_out, oh_out)
        
        true_cat_out = np.array( [1,2,3,4,0] )[None,:]
        cat_out = CharTokenizer(seq_lst = ['ACGT'], 
                                max_len = 5, 
                                alphabet = ['<pad>','A','C','G','T'])
        
        assert np.allclose(true_cat_out, cat_out)
        
        print('Encoder test passsed :)')
        
    test_encoders()
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    


    # ### Run unit tests for perfect correlation
    # def perfect_corr():
    #     # fake scores
    #     n_pre_true = torch.Tensor([1,1,1,1,1])
    #     n_post_true = torch.Tensor([1,2,3,4,5])
    #     N_pre = 1
    #     N_post = 1
        
    #     # calculate true scores
    #     _num = n_post_true/N_post
    #     _denom = n_pre_true/N_pre
    #     true_ratio = _num/_denom
    #     log_true_ratio = torch.log(true_ratio)
        
        
    #     ### Pearson Correlation test
    #     # initialization some fake scores
    #     # make pred_ratio have another dimension of 1, to match real inputs
    #     pred_ratio = torch.Tensor([1,2,3,4,5]).unsqueeze(1)
    #     log_pred_ratio = torch.log(pred_ratio)
        
    #     # test regular score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=True, corr_type=pearsonr)
    #     check_corr = corr_fn(pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==1, f'Regular score, Pearson corr: {check_corr}'
    #     del corr_fn, check_corr
        
    #     # test log score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=False, corr_type=pearsonr)
    #     check_corr = corr_fn(log_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==1, f'Log score, Pearson corr: {check_corr}'
    #     del corr_fn, check_corr

        
    #     ### Spearman Correlation test
    #     # spearman tests monotonically incr/decr
    #     scaled_pred_ratio = 3*pred_ratio
    #     scaled_log_pred_ratio = 5*log_pred_ratio
        
    #     # test regular score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=True, corr_type=spearmanr)
    #     check_corr = corr_fn(scaled_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==1, f'Regular score, Spearman corr: {check_corr}'
    #     del corr_fn, check_corr
        
    #     # test log score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=False, corr_type=spearmanr)
    #     check_corr = corr_fn(scaled_log_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==1, f'Log score, Spearman corr: {check_corr}'
        
    #     print('All correlations of 1 passed')
        
         
    # def perfect_anticorr():
    #     # fake scores
    #     n_pre_true = torch.Tensor([1,1,1,1,1])
    #     n_post_true = torch.Tensor([1,2,3,4,5])
    #     N_pre = 1
    #     N_post = 1
        
    #     # calculate true scores
    #     _num = n_post_true/N_post
    #     _denom = n_pre_true/N_pre
    #     true_ratio = _num/_denom
    #     log_true_ratio = torch.log(true_ratio)
        
        
    #     ### Pearson Correlation test
    #     # initialization some fake scores
    #     # make pred_ratio have another dimension of 1, to match real inputs
    #     pred_ratio = torch.Tensor([-1,-2,-3,-4,-5]).unsqueeze(1)
    #     log_pred_ratio = -torch.log(torch.Tensor([1,2,3,4,5])).unsqueeze(1)
        
    #     # test regular score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=True, corr_type=pearsonr)
    #     check_corr = corr_fn(pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==-1, f'Regular score, Pearson corr: {check_corr}'
    #     del corr_fn, check_corr
        
    #     # test log score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=False, corr_type=pearsonr)
    #     check_corr = corr_fn(log_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==-1, f'Log score, Pearson corr: {check_corr}'
    #     del corr_fn, check_corr

        
    #     ### Spearman Correlation test
    #     # spearman tests monotonically incr/decr
    #     scaled_pred_ratio = 3*pred_ratio
    #     scaled_log_pred_ratio = 5*log_pred_ratio
        
    #     # test regular score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=True, corr_type=spearmanr)
    #     check_corr = corr_fn(scaled_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==-1, f'Regular score, Spearman corr: {check_corr}'
    #     del corr_fn, check_corr
        
    #     # test log score check
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=False, corr_type=spearmanr)
    #     check_corr = corr_fn(scaled_log_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==-1, f'Log score, Spearman corr: {check_corr}'
        
    #     print('All correlations of -1 passed')


    # def handling_logs():
    #     # fake scores
    #     n_pre_true = torch.Tensor([1,1,1,1,1]).unsqueeze(1)
    #     n_post_true = torch.Tensor([1,2,3,4,5]).unsqueeze(1)
    #     N_pre = 1
    #     N_post = 1
        
    #     # calculate true scores
    #     _num = n_post_true/N_post
    #     _denom = n_pre_true/N_pre
    #     true_ratio = _num/_denom
    #     log_true_ratio = torch.log(true_ratio)
        
    #     ### Pearson Correlation test
    #     # initialization some fake scores
    #     # make pred_ratio have another dimension of 1, to match real inputs
    #     pred_ratio = torch.Tensor([1,2,3,4,5]).unsqueeze(1)
    #     log_pred_ratio = torch.log(pred_ratio)
        
    #     # when expOut is false, log_pred_ratio should yield correlation of 1
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=False, corr_type=pearsonr)
    #     check_corr = corr_fn(log_pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==1, f'Regular score, Pearson corr: {check_corr}'
    #     del corr_fn, check_corr
        
    #     # when expOut is true, pred_ratio should yield a correlation of 1
    #     corr_fn = CalcScoresCorr(N_pre, N_post, expOut=True, corr_type=pearsonr)
    #     check_corr = corr_fn(pred_ratio, n_pre_true, n_post_true)[-1]
    #     check_corr = round(check_corr, 3)
    #     assert check_corr==1, f'Regular score, Pearson corr: {check_corr}'
    #     del corr_fn, check_corr
    
    # # perfect_corr()
    # # perfect_anticorr()
    # handling_logs()
    