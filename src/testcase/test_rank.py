import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CUR_DIR + "/../")

import pprint as pp
import unittest
import numpy as np

from indexer import Index
from indexer import Indexable

from ranker import Rank

from searcher import IndexableResult
from searcher import Searcher

from twitter import Twitter


def sample_stop_words():
    return ['a', 'the', 'this', 'is']


class RankTests(unittest.TestCase):
    """
    Test case for Index class.
    """

    def setUp(self):
        """
        Setup ranker that will be subjected to the tests.
        """
        self.rank = Rank(sample_stop_words())

    def test_sample_ranking_with_no_exceptions(self):
        """
        Test if ranking is built without any exception.
        """
        sample1 = Indexable(1, "this is an indexable metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable metadata")
        self.rank.build_rank([sample1, sample2, sample3])

    def test_doc_frequency_matrix_with_sample1(self):
        """
        Test if document frequency matrix is correctly built.
        """
        sample1 = Indexable(1, "this is an indexable metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable metadata")
        self.rank.build_rank([sample1, sample2, sample3])

        expected_vocab_indices = {
            'an': 2, 'super': 3, 'indexable': 1, 'metadata': 0, 'another': 4
        }

        expected_tf = np.array([[1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 0, 0, 1]])

        self.assertEqual(self.rank.vocabulary, expected_vocab_indices)
        np.testing.assert_array_equal(self.rank.ft_matrix.todense(),
                                      expected_tf)

    def test_doc_frequency_matrix_with_sample2(self):
        """
        Test if document frequency matrix is correctly built.
        """
        sample1 = Indexable(1, "the sky is blue")
        sample2 = Indexable(2, "the sun is bright")
        self.rank.build_rank([sample1, sample2])

        expected_vocab_indices = {'blue': 0, 'sun': 2, 'bright': 3, 'sky': 1}

        expected_tf = np.array([[1, 1, 0, 0],
                                [0, 0, 1, 1]])

        self.assertEqual(self.rank.vocabulary, expected_vocab_indices)
        np.testing.assert_array_equal(self.rank.ft_matrix.todense(), expected_tf)

    def test_doc_inverse_term_frequency_vector1(self):
        """
        Test if document inverse term frequency vector is correctly built.
        """
        sample1 = Indexable(1, "this is an indexable metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable metadata")
        self.rank.build_rank([sample1, sample2, sample3])

        expected_idf = [1.,  1., 1.28768207, 1.69314718, 1.69314718]
        expected_tf_idf = [[0.52284231, 0.52284231, 0.67325467, 0, 0],
                           [0.39148397, 0.39148397, 0.50410689, 0.66283998, 0],
                           [0.45329466, 0.45329466, 0, 0, 0.76749457]]

        np.testing.assert_almost_equal(
            self.rank.ifd_diag_matrix.diagonal(), expected_idf, 4)

        np.testing.assert_almost_equal(
            self.rank.tf_idf_matrix.todense(), expected_tf_idf, 4)

    def test_doc_inverse_term_frequency_vector2(self):
        """
        Test if document inverse term frequency vector is correctly built.
        """
        sample1 = Indexable(1, "the sky is blue")
        sample2 = Indexable(2, "the sun is bright")
        self.rank.build_rank([sample1, sample2])

        expected_idf = [1.40546511, 1.40546511, 1.40546511, 1.40546511]
        expected_tf_idf = [[0.70710678, 0.70710678, 0, 0],
                           [0, 0, 0.70710678, 0.70710678]]

        np.testing.assert_almost_equal(
            self.rank.ifd_diag_matrix.diagonal(), expected_idf, 4)

        np.testing.assert_almost_equal(
            self.rank.tf_idf_matrix.todense(), expected_tf_idf, 4)

    def test_score_computation(self):
        """
        Test if document score is correctly calculated.
        """
        sample1 = Indexable(1, "the sky is blue")
        self.rank.build_rank([sample1])

        np.testing.assert_almost_equal(
            self.rank.compute_rank(0, ["blue"]), 0.707106, 5)
        np.testing.assert_almost_equal(
            self.rank.compute_rank(0, ["sky"]), 0.7071067, 5)
        np.testing.assert_almost_equal(
           self.rank.compute_rank(0, ["blue", "sky"]), 1.414213, 5)

    def test_debug_ft_matrix(self):
        self.twitter = Twitter(CUR_DIR + "/test_crossfit.tweets", CUR_DIR + "/test_stop_words.txt")
        self.twitter.load_tweets_and_build_index()

        

if __name__ == '__main__':
    unittest.main()