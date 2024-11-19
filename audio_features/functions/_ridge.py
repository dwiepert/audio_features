"""
Ridge regression functions

Author(s): Aditya Vaidya, Daniela Wiepert
Last modified: 11/13/2024
"""
#IMPORTS
#built-in
import itertools as itools
import json
import logging
import random 
import socket
from pathlib import Path
import tempfile
from typing import List, Union 

#third-party
import numpy as np
import torch
from tqdm import tqdm



def _zscore(mat, return_unzvals=False):
    """Z-scores the rows of [mat] by subtracting off the mean and dividing
    by the standard deviation.
    If [return_unzvals] is True, a matrix will be returned that can be used
    to return the z-scored values to their original state.

    :param mat: array like, matrix
    :return zmat: array like, z-scored matrix
    """
    zmat = np.empty(mat.shape, mat.dtype)
    unzvals = np.zeros((zmat.shape[0], 2), mat.dtype)
    for ri in range(mat.shape[0]):
        unzvals[ri,0] = np.std(mat[ri,:])
        unzvals[ri,1] = np.mean(mat[ri,:])
        zmat[ri,:] = (mat[ri,:]-unzvals[ri,1]) / (1e-10+unzvals[ri,0])
    
    if return_unzvals:
        return zmat, unzvals
    
    return zmat

def _mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    :param d: 1D (N,) array (contains the diagonal elements)
    :param mtx: 2D (N,N) array
    :param left: bool, default True

    :return: mult_diag(d, mts, left=True) == dot(diag(d), mtx)
             mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx
    
def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, logger=logging.getLogger("ridge_corr"), solver_dtype=None):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    :param stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    :param resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    :param alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    :param normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.

    :return wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        U,S,Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError:
        raise NotImplementedError('There was an np.linalg.LinAnlgError but no way to work with this')
        #logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        #from text.regression.svd_dgesvd import svd_dgesvd
        #U,S,Vh = svd_dgesvd(stim, full_matrices=False)

    UR = U.T @ np.nan_to_num(resp)

    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float,int)):
        alpha = np.ones(resp.shape[1]) * alpha

    # Convert to desired dtype after SVD
    if solver_dtype is not None:
        UR, S, Vh, alpha = (x.astype(solver_dtype) for x in [UR, S, Vh, alpha])

    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]), dtype=Vh.dtype)
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        #awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        D = S/(S**2+ua**2)
        #if Vh.T.shape[0] > selvox.shape[0]:
        #    awt = Vh.T @ (D[:, None] * UR[:, selvox])
        #else:
        #    awt = (Vh.T * D) @ UR[:, selvox]
        wt[:,selvox] = (Vh.T * D) @ UR[:, selvox]
    return wt

def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=logging.getLogger("ridge_corr"), solver_dtype=np.float32):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    :param Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    :param Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    :param Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    :param Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    :param alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    :param normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    :param corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    :param singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    :param use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    :return Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.

    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        raise NotImplementedError('There was an np.linalg.LinAnlgError but no way to work with this')
        #logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        #from text.regression.svd_dgesvd import svd_dgesvd
        #U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)
    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = U.T @ Rresp ## Precompute this matrix product for speed
    PVh = Pstim @ Vh.T ## Precompute this matrix product for speed

    # Convert to desired dtype after SVD
    if solver_dtype is not None:
        UR, S, PVh, alphas = (x.astype(solver_dtype) for x in [UR, S, PVh, alphas])

    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = _zscore(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, tqdm(alphas, desc='alphas', leave=False)):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter
        D = S / (S ** 2 + na ** 2) ## Reweight singular vectors by the (normalized?) ridge parameter

        pred = _mult_diag(D, PVh, left=False) @ UR ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)

        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)

        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * _zscore(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)

        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        logger.info(log_msg)

    return Rcorrs

def residuals(resp, wt):
    """
    Generate residuals of regression

    :param resp: ground truth response to predict
    :param wt: linear regression weights
    :return: the residual (predicted response - true response)
    """
    resp = resp.astype(wt.dtype)
    pred = (resp @ wt)
    return np.subtract(pred,resp)

def bootstrap_ridge(Rstim, Rresp, alphas, nboots, chunklen, nchunks, corrmin=0.2, joined=None, 
                    singcutoff=1e-10, normalpha=False, single_alpha=False,use_corr=True, solver_dtype=None,
                    logger=logging.getLogger("ridge_corr")):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.

    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist
    will be given the same regularization parameter (the one that is the best on average).

    Parameters
    ----------
    :param Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. (Each feature should be Z-scored across time - function handles this)
    :param Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    :param alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    :param nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    :param chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    :param nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this
        product should be about 20 percent of the total length of the training data.
    :param corrmin : float in [0..1], default 0.2
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    :param joined : None or list of array_like indices, default None
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    :param singcutoff : float, default 1e-10
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    :param normalpha : boolean, default False
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    :param single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    :param use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    

    :return wt : array_like, shape (N, M)
        regression weights for N features and M responses. 
    :return r: array_like (TR, M)
    """

    #Rstim = _zscore(Rstim)
    #Rresp = _zscore(Rresp)

    nresp, nvox = Rresp.shape
    valinds = [] # Will hold the indices into the validation data for each bootstrap

    Rcmats = []
    for bi in tqdm(range(nboots), desc='Ridge bootstraps', leave=False):
        logger.info("Selecting held-out test set..")
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        valinds.append(heldinds)

        RRstim = Rstim[notheldinds,:]
        PRstim = Rstim[heldinds,:]
        RRresp = Rresp[notheldinds,:]
        PRresp = Rresp[heldinds,:]

        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                           corrmin=corrmin, singcutoff=singcutoff,
                           normalpha=normalpha, use_corr=use_corr,
                           logger=logger, solver_dtype=solver_dtype)

        Rcmats.append(Rcmat)

    #Find best alphas
    if nboots>0:
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None

    if not single_alpha:
        if nboots==0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")

        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = allRcorrs[:,jl,:].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        if nboots==0:
            if len(alphas)==1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = allRcorrs.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]

        valphas = np.array([bestalpha]*nvox)
        logger.info("Best alpha = %0.3f"%bestalpha)
    
    ## GET RESIDUALS
    wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha, solver_dtype=solver_dtype)

    return wt, valphas

class RidgeRegression():
    def __init__(self, stim, resp, save_path:Union[str,Path], cci_features=None, fnames:List=[],
                 nboots=50, chunklen=40, nchunks=125, singcutoff=1e-10,
			    use_corr=False, single_alpha=False, alphas_logspace=(1,4,10), overwrite:bool=False, local_path:Union[str,Path]=None):
        self.stim = _zscore(stim) #EXPECTS CONCATENATED VECTORS 
        self.resp = _zscore(resp)
        self.save_path=Path(save_path)
        self.cci_features = cci_features
        if self.cci_features is None:
            self.local_path = self.save_path
        else:
            assert self.local_path is not None, 'Please give a local path to save config files to (json not cotton candy compatible)'

        self.nboots=nboots
        self.chunklen=chunklen
        self.nchunks=nchunks
        self.singcutoff=singcutoff
        self.use_corr=use_corr
        self.single_alpha=single_alpha 

        alphas_log_min, alphas_log_max, num_alphas = alphas_logspace
        self.alphas = np.logspace(alphas_log_min, alphas_log_max, num_alphas)

        self.config = {
			'nboots': nboots, 'chunklen': chunklen,
			'nchunks': nchunks, 'singcutoff': singcutoff, 'use_corr': use_corr,
			'single_alpha': single_alpha, 'alphas': self.alphas.tolist()}
        
        with open(str(self.local_path / 'RidgeRegression_config.json'), 'w') as f:
            json.dump(self.config)

        self.overwrite=overwrite

        #TODO: SAVE PATHS EDITING
        #TODO: create a tempfile and upload configs to the save path
        self.result_paths = {'weights': self.save_path /'weights',
                            'valphas': self.save_path / 'valphas'}

        self.result_paths['residuals'] = {}
        r_path = self.save_path / 'residuals'
        for f in fnames:
            self.result_paths['residuals'][f] = r_path / f
        
        self._check_previous()

    def _check_previous(self):
        self.weights_exist = False
        self.valphas_exist = False

        if self.cci_features is not None:
            if self.cci_features.exists_object(str(self.save_path)):
                if self.cci_features.exists_object(str(self.results_paths['weights'])): self.weights_exist=True
                if self.cci_features.exists_object(str(self.results_paths['valpha'])): self.valphas_exist=True 
            else:

                if Path(str(self.results_paths['weights'])+'.npz').exists(): self.weights_exist=True
                if Path(str(self.results_paths['valphas'])+'.npz'): self.valphas_exist = True

        if self.weights_exist and not self.overwrite: 
            if self.cci_features is not None:
                self.wt = self.cci_features.download_raw_array(str(self.results_paths['weights']))
            else:
                self.wt = np.load(str(self.results_paths['weights'])+'.npz')
        else:
            self.wt = None 
            


    def run_regression(self, prev_valphas=None):
        """
        """
        if self.wt is not None and not self.overwrite:
            print('Weights already exist and should not be overwritten')
            return
        
        if self.valphas_exist and not self.overwrite:
            if self.cci_features is not None:
                valphas = self.cci_features.download_raw_array(str(self.results_paths['valphas']))
            else:
                valphas = np.load(str(self.results_paths['valphas'])+'.npz')
            return self.run_regression(prev_valphas=valphas)
        
        if prev_valphas is not None:
            self.wt = ridge(stim=self.stim, resp=self.resp, alpha=prev_valphas, singcutoff=self.singcutoff, normalalpha=False, solver_dtype=np.float32)
        else:
            self.wt, valphas = bootstrap_ridge(Rstim=self.stim, Rresp=self.resp, alphas=self.alphas, nboots=self.nboots,
                                            chunklen=self.chunklen, nchunks=self.nchunks, singcutoff=self.singcutoff, 
                                            single_alpha=self.single_alpha, use_cor=self.use_corr, solver_dtype=np.float32)

        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['weights'], self.wt)
            self.cci_features.upload_raw_array(self.result_paths['valphas'], valphas)
        else:
            np.savez_compressed(str(self.result_paths['weights']+'.npz'), self.wt)
            np.savez_compressed(str(self.result_paths['valphas']+'.npz'), valphas)
        
        return

    def extract_residuals(self, feats, fname):
        """
        """
        if self.cci_features is not None:
            if self.cci_features.exists_object(self.result_paths['residuals'][fname]) and not self.overwrite:
                return 
        else:
            if Path(str(self.result_paths['residuals'][fname]) + '.npz').exists() and not self.overwrite:
                return 
            
        assert self.wt is not None, 'Regression has not been run yet. Please do so.'
        r = residuals(feats, self.wt)

        if self.cci_features:
            self.cci_features.upload_raw_array(self.result_paths['residuals'][fname], residuals)
        else:
            np.savez_compressed(str(self.result_paths['residuals'][fname])+'.npz', residuals)