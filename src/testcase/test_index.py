# -*- coding: utf-8 -*-
import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CUR_DIR + "/../")

import unittest

from collections import defaultdict

from indexer import Index
from indexer import Indexable

from ranker import Rank

from searcher import IndexableResult
from searcher import Searcher

from twitter import Twitter

def sample_stop_words():
    return ['a', 'the', 'this', 'is', 'between']


class IndexerTests(unittest.TestCase):
    """
    Test case for Index class.
    """

    def setUp(self):
        """
        Setup index that will be subjected to the tests.
        """
        self.index = Index(sample_stop_words())

    def test_sample_index_words_count(self):
        sample = Indexable(1, "this is an indexable metadata, that is an indexable super metadata")

        expected_words_count = defaultdict(int)
        expected_words_count['this'] = 1
        expected_words_count['that'] = 1
        expected_words_count['super'] = 1
        expected_words_count['is'] = 2
        expected_words_count['metadata,'] = 1 # Exact terms are not yet processed.
        expected_words_count['metadata'] = 1
        expected_words_count['indexable'] = 2
        expected_words_count['an'] = 2

        self.assertItemsEqual(sample.words_count, expected_words_count)

    def test_sample_indexing_and_validate_items(self):
        sample1 = Indexable(1, "this is an indexable metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable metadata")

        self.index.build_index([sample1, sample2, sample3])

        expected_term_index = defaultdict(int)
        expected_term_index['indexable'] = [0, 1, 2]

        self.assertItemsEqual(self.index.term_index['indexable'], expected_term_index['indexable'])

    def test_invalid_term_search(self):
        """
        Test if the search returns when the term is not found.
        """
        sample1 = Indexable(1, "this is an indexable simple metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable metadata")

        expected_indices = []

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(["not_valid_term"])

        self.assertItemsEqual(search_results, expected_indices)

    def test_mixed_valid_invalid_term_search(self):
        """
        Test if the search returns when there are valid and invalid terms mixed.
        """
        sample1 = Indexable(1, "this is an indexable simple metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable metadata")

        expected_indices = []

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(["not_valid_term", "super"])

        self.assertItemsEqual(search_results, expected_indices)

    def test_one_term_search(self):
        """
        Test if the search for one term returns expected results.
        """
        sample1 = Indexable(1, "this is an indexable metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable super metadata")

        expected_indices = [1, 2]

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(["super"])

        self.assertItemsEqual(search_results, expected_indices)

    def test_stop_word_search(self):
        """
        Test if stop words are correctly ignored.
        """
        sample1 = Indexable(1, "this is an indexable metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable super metadata")

        expected_indices = []

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(["this"])

        self.assertItemsEqual(search_results, expected_indices)

    def test_two_terms_search(self):
        """
        Test if the search for two term returns expected results.
        """
        sample1 = Indexable(1, "this is an indexable simple metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable super metadata")

        expected_indices = [1, 2]

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(["indexable", "super"])

        self.assertItemsEqual(search_results, expected_indices)

    def test_three_terms_search_with_stop_words(self):
        """
        Test if the search for stop words returns expected results.
        """
        sample1 = Indexable(1, "this is an indexable simple metadata")
        sample2 = Indexable(2, "this is an indexable super metadata")
        sample3 = Indexable(3, "this is another indexable super metadata")

        expected_indices = [0, 1, 2]

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(["this", "is", "metadata"])

        self.assertItemsEqual(search_results, expected_indices)        


if __name__ == '__main__':
    unittest.main()
