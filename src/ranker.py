# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import scipy.sparse.sparsetools as sptools
import logging
import time

def timed(fn):
    """Decorator used to benchmark functions runtime.

    """

    def wrapped(*arg, **kw):
        ts = time.time()
        result = fn(*arg, **kw)
        te = time.time()

        logging.info("[Ranker] Function = %s, Elapsed Time = %2.2f sec" \
              % (fn.__name__, (te - ts)))

        return result

    return wrapped
    
class Rank(object):
    """Class encapsulating tf-idf ranking logic.

    Tf-idf means stands for term-frequency times inverse document-frequency
    and it is a common term weighting scheme in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    Args:
      smoothing (int, optional): Smoothing parameter for tf-idf computation
        preventing by-zero divisions when a term does not occur in corpus.

      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    Attributes:
      smoothing (int, optional): Smoothing parameter for tf-idf computation
        preventing by-zero divisions when a term does not occur in corpus.

      stop_words (list of str): Stop words that will be filtered during docs
        processing.

      vocabulary (dict): Dictionary containing unique words of the corpus as
        keys and their respective global index used in tf-idf data structures.

      ft_matrix (matrix): Matrix containing the frequency term for each term
        in the corpus respecting the index stored in `vocabulary`.

      ifd_diag_matrix (list): Vector containing the inverse document
        frequency for each term in the corpus. It respects the index stored in
        `vocabulary`.

      tf_idf_matrix (matrix): Matrix containing the ft-idf score for each term
        in the corpus respecting the index stored in `vocabulary`.

    """

    def __init__(self, stop_words, smoothing=1):
        self.smoothing          = smoothing
        self.stop_words         = stop_words
        self.vocabulary         = {}
        self.ft_matrix          = []
        self.ifd_diag_matrix    = []
        self.tf_idf_matrix      = []

    def build_rank(self, tweets):
        """Build tf-idf ranking score for terms in the corpus.

        Note:
          The code in this method could have been extracted to other smaller
          methods, improving legibility. This extraction has not been done so
          that its runtime complexity can be computed easily (the runtime
          complexity can be improved).

        Args:
          tweets (list of Indexable): List of indexed tweets that will be
            considered during tf-idf score computation.

        """
        self.__build_vocabulary(tweets)

        n_terms = len(self.vocabulary)
        n_docs = len(tweets)
        ft_matrix = sp.lil_matrix((n_docs, n_terms), dtype=np.dtype(float))

        logging.info("[Ranker] Vocabulary assembled with terms count %s, docs count %s" \
            % ("{:,}".format(n_terms), "{:,}".format(n_docs)))

        # compute tf
        logging.info("[Ranker] Starting tf computation ...")
        for index, indexable in enumerate(tweets):
            for word in indexable.words_generator(self.stop_words):
                word_index_in_vocabulary = self.vocabulary[word]
                doc_word_count = indexable.count_for_word(word)
                ft_matrix[index, word_index_in_vocabulary] = doc_word_count
        # return a copy of this matrix in compressed sparse column format.
        self.ft_matrix = ft_matrix.tocsc()

        # compute idf with smoothing
        logging.info("[Ranker] Starting tf-idf computation ...")
        df = np.diff(self.ft_matrix.indptr) + self.smoothing
        n_docs_smooth = n_docs + self.smoothing

        # create diagonal matrix to be multiplied with ft
        idf = np.log(float(n_docs_smooth) / df) + 1.0
        self.ifd_diag_matrix = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)

        # compute tf-idf
        self.tf_idf_matrix = self.ft_matrix * self.ifd_diag_matrix
        self.tf_idf_matrix = self.tf_idf_matrix.tocsr()

        # compute td-idf normalization
        norm = self.tf_idf_matrix.tocsr(copy=True)
        norm.data **= 2
        norm = norm.sum(axis=1)
        n_nzeros = np.where(norm > 0)
        norm[n_nzeros] = 1.0 / np.sqrt(norm[n_nzeros])
        norm = np.array(norm).T[0]
        sptools.csr_scale_rows(self.tf_idf_matrix.shape[0],
                                      self.tf_idf_matrix.shape[1],
                                      self.tf_idf_matrix.indptr,
                                      self.tf_idf_matrix.indices,
                                      self.tf_idf_matrix.data, norm)

    def __build_vocabulary(self, tweets):
        """Build vocabulary with indexable tweets.

        Args:
          tweets (list of Indexable): Indexed tweets that will be
            considered during ranking.

        """
        vocabulary_index = 0
        for indexable in tweets:
          for word in indexable.words_generator(self.stop_words):
              if word not in self.vocabulary:
                  self.vocabulary[word] = vocabulary_index
                  vocabulary_index += 1

    def compute_rank(self, doc_index, terms):
        """Compute tf-idf score of an indexed document.

        Args:
          doc_index (int): Index of the document to be ranked.
          terms (list of str): List of query terms.

        Returns:
          float: tf-idf of document identified by its index.

        """
        score = 0
        for term in terms:
            term_index = self.vocabulary[term]
            score += self.tf_idf_matrix[doc_index, term_index]
        return score
