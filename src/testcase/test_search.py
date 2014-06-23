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


class SearcherTests(unittest.TestCase):
    """
    Test case for SearchEngine class.
    """

    def setUp(self):
        """
        Setup search engine that will be subjected to the tests.
        """
        self.twitter = Twitter(CUR_DIR + "/test_crossfit.tweets", CUR_DIR + "/test_stop_words.txt")
        self.twitter.load_tweets_and_build_index()

        self.searcher = Searcher(self.twitter.tweets, self.twitter.stop_words)

    def test_indexed_doc_count(self):

        self.assertEqual(self.searcher.count(), 10)

    def test_existent_term_search(self):
        """
        Test if search is correctly performed.
        """
        results = self.searcher.search("coach")
        expected_results = 3

        self.assertEqual(results[0].indexable.docid, expected_results)

    def test_non_existent_term_search(self):
        """
        Test if search is correctly performed.
        """

        expected_results = []
        results = self.searcher.search("asdasdasdas")

        self.assertListEqual(results, expected_results)

    def test_search_result_limit(self):
        """
        Test if search results can be limited.
        """
        results = self.searcher.search("crossfit", 1)
        expected_results = 6

        self.assertEqual(results[0].indexable.docid, expected_results)


if __name__ == '__main__':
    unittest.main()