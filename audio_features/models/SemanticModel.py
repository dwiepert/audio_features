"""
Semantic model for extraction word identity vectors

Author(s): Alexander Huth
Last modified: Unknown
"""
#IMPORTS
##built-in
import logging
from typing import List

##third-party
import tables
import numpy as np

logger = logging.getLogger("SemanticModel")

class SemanticModel(object):
    """This class defines a semantic vector-space model based on HAL or LSA with some
    prescribed preprocessing pipeline.
    
    It contains two important variables: vocab and data.
    :param vocab: List, a 1D list (or array) of words.
    :param data: is a 2D array (features by words) of word-feature values.
    """
    def __init__(self, data, vocab):
        """Initializes a SemanticModel with the given [data] and [vocab].
        """
        self.data = data
        self.vocab = vocab
    
    def get_ndim(self):
        """
        :return: int, number of dimensions in this model.
        """
        return self.data.shape[0]
    ndim = property(get_ndim)

    def get_vindex(self):
        """
        :return _vindex: doct, {vocab: index} dictionary.
        """
        if "_vindex" not in dir(self):
            self._vindex = dict([(v,i) for (i,v) in enumerate(self.vocab)])
        return self._vindex
    vindex = property(get_vindex)

    def __getitem__(self, word:str):
        """
        :param word: str, word
        :return: array-like, vector corresponding to the given [word].
        """
        return self.data[:,self.vindex[word]]
    
    def load_root(self, rootfile:str, vocab:list):
        """
        Load the SVD-generated semantic vector space from [rootfile], assumed to be
        an HDF5 file.

        :param rootfile: str, root file, hdf5
        :param vocab: List, vocab list
        """
        roothf = tables.openFile(rootfile)
        self.data = roothf.getNode("/R").read()
        self.vocab = vocab
        roothf.close()

    def load_ascii_root(self, rootfile:str, vocab:list):
        """
        Loads the SVD-generated semantic vector space from [rootfile], assumed to be
        an ASCII dense matrix output from SDVLIBC.

        :param rootfile: str, root file, ascii dense matrix output
        :param vocab: list, vocab list
        """
        vtfile = open(rootfile)
        nrows, ncols = map(int, vtfile.readline().split())
        Vt = np.zeros((nrows,ncols))
        nrows_done = 0
        for row in vtfile:
            Vt[nrows_done,:] = map(float, row.split())
            nrows_done += 1

        self.data = Vt
        self.vocab = vocab
    
    def restrict_by_occurrence(self, min_rank:int=60, max_rank:int=60000):
        """
        Restricts the data to words that have an occurrence rank lower than
        [min_rank] and higher than [max_rank].

        :param min_rank: int, minimum occcurence rank
        :param max_rank: int, maximum occurrence rank
        """
        logger.debug("Restricting words by occurrence..")
        nwords = self.data.shape[1]
        wordranks = np.argsort(np.argsort(self.data[0,:]))
        goodwords = np.nonzero(np.logical_and((nwords-wordranks)>min_rank,
                                              (nwords-wordranks)<max_rank))[0]

        self.data = self.data[:,goodwords]
        self.vocab = [self.vocab[i] for i in goodwords]
        logger.debug("Done restricting words..")

    def pca_reduce(self, ndims:int):
        """
        Reduces the dimensionality of the vector-space using PCA.
        :param ndims: int, PCA dims to keep
        """
        logger.debug("Reducing with PCA to %d dimensions"%ndims)
        U,S,Vh = np.linalg.svd(self.data, full_matrices=False)
        self.data = np.dot(Vh[:ndims].T, np.diag(S[:ndims])).T
        logger.debug("Done with PCA..")

    def pca_reduce_multi(self, ndimlist:List[int]):
        """Reduces the dimensionality of the vector-space using PCA for many
        different numbers of dimensions.  More efficient than running
        pca_reduce many times.
        
        Instead of modifying this object, this function returns a list of new
        SemanticModels with the specified numbers of dimensions.

        :param ndimlist: List, list of dimensions to keep for many different diimensions
        :return newmodels: List, list of new SemanticModels with specified ndims
        """
        logger.debug("Reducing with PCA to fewer dimensions..")
        U,S,Vh = np.linalg.svd(self.data, full_matrices=False)
        newmodels = []
        for nd in ndimlist:
            newmodel = SemanticModel()
            newmodel.vocab = list(self.vocab)
            newmodel.data = np.dot(Vh[:nd].T, np.diag(S[:nd])).T
            newmodels.append(newmodel)
        return newmodels
    
    def save(self, filename:str):
        """
        Saves this semantic model at the given filename.
        :param filename: str, filename to save to
        """
        logger.debug("Saving file: %s"%filename)
        shf = tables.openFile(filename, mode="w", title="SemanticModel")
        shf.createArray("/", "data", self.data)
        shf.createArray("/", "vocab", self.vocab)
        shf.close()
        logger.debug("Done saving file..")

    @classmethod
    def load(cls, filename:str):
        """
        Loads a semantic model from the given filename.
        :param cls: SemanticModel
        :param filename: str, filename to load from
        """
        logger.debug("Loading file: %s"%filename)
        shf = tables.open_file(filename)

        newsm = cls(None, None)
        newsm.data = shf.get_node("/data").read()
        newsm.vocab = [s.decode('utf-8') for s in shf.get_node("/vocab").read()]
        
        shf.close()
        logger.debug("Done loading file..")
        return newsm
    
    def copy(self):
        """
        Returns a copy of this model.
        :return cp: SemanticModel, copy of this model
        """
        logger.debug("Copying model..")
        cp = SemanticModel(self.data.copy(), list(self.vocab))
        logger.debug("Done copying model..")
        return cp

    def project_stims(self, stimwords:list):
        """
        Projects the stimuli given in [stimwords], which should be a list of lists
        of words, into this feature space. Returns the average feature vector across
        all the words in each stimulus.
        
        :param stimwords: List, List of lists of words
        :return pstim: List, average feature vector across all the words in each stimulus.
        """
        logger.debug("Projecting stimuli..")
        stimlen = len(stimwords)
        ndim = self.data.shape[0]
        pstim = np.zeros((stimlen, ndim))
        vset = set(self.vocab)
        for t in range(stimlen):
            dropped = 0
            for w in stimwords[t]:
                dropped = 0
                if w in vset:
                    pstim[t] += self[w]
                else:
                    dropped += 1
            
            pstim[t] /= (len(stimwords[t])-dropped)

        return pstim

    def uniformize(self):
        """
        Uniformizes each feature.
        """
        logger.debug("Uniformizing features..")
        R = np.zeros_like(self.data).astype(np.uint32)
        for ri in range(self.data.shape[0]):
            R[ri] = np.argsort(np.argsort(self.data[ri]))
        
        self.data = R.astype(np.float64)
        logger.debug("Done uniformizing...")

    def gaussianize(self):
        """
        Gaussianizes each feature.
        """
        logger.debug("Gaussianizing features..")
        self.data = gaussianize_mat(self.data.T).T
        logger.debug("Done gaussianizing..")

    def zscore(self, axis:int=0):
        """
        Z-scores either each feature (if axis is 0) or each word (if axis is 1).
        If axis is None nothing will be Z-scored.

        :param axis: int, axis to zscore
        """
        if axis is None:
            logger.debug("Not Z-scoring..")
            return
        
        logger.debug("Z-scoring on axis %d"%axis)
        if axis==1:
            self.data = zscore(self.data.T).T
        elif axis==0:
            self.data = zscore(self.data)
    
    def rectify(self):
        """
        Rectifies the features.
        """
        self.data = np.vstack([-np.clip(self.data, -np.inf, 0), np.clip(self.data, 0, np.inf)])
    
    def clip(self, sds:int):
        """
        Clips feature values more than [sds] standard deviations away from the mean
        to that value.  Another method for dealing with outliers.

        :param sds: int, standard deviations to keep
        """
        logger.debug("Truncating features to %d SDs.."%sds)
        fsds = self.data.std(1)
        fms = self.data.mean(1)
        newdata = np.zeros(self.data.shape)
        for fi in range(self.data.shape[0]):
            newdata[fi] = np.clip(self.data[fi],
                                  fms[fi]-sds*fsds[fi],
                                  fms[fi]+sds*fsds[fi])

        self.data = newdata
        logger.debug("Done truncating..")

    def find_words_like_word(self, word:str,  n:int=10):
        """
        Finds the [n] words most like the given [word].
        :param word: str, word
        :param n: int, number of words to look for
        :return: List[str], n words
        """
        return self.find_words_like_vec(self.data[:,self.vocab.index(word)], n)

    def find_words_like_vec(self, vec:np.ndarray, n:int=10, corr:bool=True):
        """
        Finds the [n] words most like the given [vector].
        
        :param vec: np.ndarray
        :param n: int, number of similar words to look for
        :param corr: bool, true if using correlation
        :return words: List[str], words (n dims)
        """
        nwords = len(self.vocab)
        if corr:
            corrs = np.nan_to_num([np.corrcoef(vec, self.data[:,wi])[1,0] for wi in range(nwords)])
            scorrs = np.argsort(corrs)
            words = list(reversed([(corrs[i], self.vocab[i]) for i in scorrs[-n:]]))
        else:
            proj = np.nan_to_num(np.dot(vec, self.data))
            sproj = np.argsort(proj)
            words = list(reversed([(proj[i], self.vocab[i]) for i in sproj[-n:]]))
        return words

    def find_words_like_vecs(self, vecs:List[np.ndarray], n:int=10, corr:bool=True, distance_cull:int=None):
        """
        ind the `n` words most like each vector in `vecs`.

        :param vecs: list of np.ndarray vectors
        :param n: int, number of similar words to look for
        :param corr: bool, true if using correlation
        :param distance_cull: int, number of closest neighbors to remove
        :return: np.ndarray, word vectors
        """
        if corr:
            from database_utils.functions import xcorr
            vproj = xcorr(vecs, self.data.T)
        else:
            vproj = np.dot(vecs, self.data)

        return np.vstack([self._get_best_words(vp, n, distance_cull) for vp in vproj])

    def _get_best_words(self, proj:np.ndarray, n:int=10, distance_cull:int=None):
        """
        Find the `n` words corresponding to the highest values in the vector `proj`.
        If `distance_cull` is an int, greedily find words with the following algorithm:
        1. Initialize the possible set of words with all words.
        2. Add the best possible word, w*. Remove w* from the possible set.
        3. Remove the `distance_cull` closest neighbors of w* from the possible set.
        4. Goto 2.

        :param proj: np.ndarray, projection vector
        :param n: int, number of similar words to look fo
        :param distance_cull: int, number of closest neighbors to remove
        :return best_words: np.ndarray, best words
        """
        vocarr = np.array(self.vocab)
        if distance_cull is None:
            return vocarr[np.argsort(proj)[-n:][::-1]]
        elif not isinstance(distance_cull, int):
            raise TypeError("distance_cull should be an integer value, not %s" % str(distance_cull))

        poss_set = set(self.vocab)
        poss_set = np.arange(len(self.vocab))
        best_words = []
        while len(best_words) < n:
            # Find best word in poss_set
            best_poss = poss_set[proj[poss_set].argmax()]
            # Add word to best_words
            best_words.append(self.vocab[best_poss])
            # Remove nearby words (by L2-norm..?)
            bwdists = ((self.data.T - self.data[:,best_poss])**2).sum(1)
            nearest_inds = np.argsort(bwdists)[:distance_cull+1]
            poss_set = np.setdiff1d(poss_set, nearest_inds)

        return np.array(best_words)
    
    def similarity(self, word1:np.ndarray, word2:np.ndarray):
        """
        Returns the correlation between the vectors for [word1] and [word2].

        :param word1: np.ndarray, vector representation of word 1
        :param word2: np.ndarray, vector representation of word 2
        :return: correlation between word1 and word2
        """
        return np.corrcoef(self.data[:,self.vocab.index(word1)], self.data[:,self.vocab.index(word2)])[0,1]

    def print_best_worst(self, ii:int, n:int=10):
        """
        Print the best and worst 

        :return ii: int, index of row to look in
        :return n: int, number of words to look for
        """
        vector = self.data[ii]
        sv = np.argsort(self.data[ii])
        print ("Best:")
        print ("-------------")
        for ni in range(1,n+1):
            print ("%s: %0.08f"%(np.array(self.vocab)[sv[-ni]], vector[sv[-ni]]))
            
        print ("\nWorst:")
        print ("-------------")
        for ni in range(n):
            print ("%s: %0.08f"%(np.array(self.vocab)[sv[ni]], vector[sv[ni]]))
            
        print ("\n")


def gaussianize(vec:np.ndarray):
    """
    Uses a look-up table to force the values in [vec] to be gaussian.

    :param vec: np.ndarray, word vector
    :return zvals: np.ndarray, gaussian vals for vec
    """
    import scipy.stats
    ranks = np.argsort(np.argsort(vec))
    cranks = (ranks+1).astype(float)/(ranks.max()+2)
    vals = scipy.stats.norm.isf(1-cranks)
    zvals = vals/vals.std()
    return zvals

def gaussianize_mat(mat:np.ndarray):
    """
    Gaussianizes each column of [mat].

    :param mat: np.ndarray, matrix
    :Return gmat: np.ndarray, gaussianized matrix
    """
    gmat = np.empty(mat.shape)
    for ri in range(mat.shape[1]):
        gmat[:,ri] = gaussianize(mat[:,ri])
    return gmat

def zscore(mat:np.ndarray, return_unzvals:bool=False):
    """
    Z-scores the rows of [mat] by subtracting off the mean and dividing
    by the standard deviation.
    If [return_unzvals] is True, a matrix will be returned that can be used
    to return the z-scored values to their original state.

    :param mat: np.ndarray matrix
    :param return_unzvals: bool, true if returning the values for reversing
    :return zmat: np.ndarray, zscored matrix
    :return unzvals: np.ndarray, vals to unzscore the matrix
    """
    zmat = np.empty(mat.shape)
    unzvals = np.zeros((zmat.shape[0], 2))
    for ri in range(mat.shape[0]):
        unzvals[ri,0] = np.std(mat[ri,:])
        unzvals[ri,1] = np.mean(mat[ri,:])
        zmat[ri,:] = (mat[ri,:]-unzvals[ri,1]) / (1e-10+unzvals[ri,0])
    
    if return_unzvals:
        return zmat, unzvals
    
    return zmat
