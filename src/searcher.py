# -*- coding: utf-8 -*-
import os
import logging
import pickle
import re
import time

from collections import defaultdict

from ranker import Rank

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def timed(fn):
    """Decorator used to benchmark functions runtime.

    """

    def wrapped(*arg, **kw):
        ts = time.time()
        result = fn(*arg, **kw)
        te = time.time()

        logging.info("[Searcher] Function = %s, Elapsed Time = %2.2f sec" \
              % (fn.__name__, (te - ts)))

        return result

    return wrapped

class IndexableResult(object):
    """Class representing a search result with a tf-idf score.

    Args:
      score (float): tf-idf score for the result.
      indexable (Indexable): Indexed object.

    Attributes:
      score (float): tf-idf score for the result.
      indexable (Indexable): Indexed object.

    """

    def __init__(self, score, indexable):
        self.score = score
        self.indexable = indexable

    def __repr__(self):
        return "score: %f, %s" % (self.score, self.indexable)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and abs(self.score - other.score) < 0.0001
                and self.indexable == other.indexable)

    def __ne__(self, other):
        return not self.__eq__(other)


class Searcher(object):
    """Search engine for objects that can be indexed.

    Attributes:
      objects (list of Indexable): List of objects that can be considered
        during search.
      stop_words (list of str): Stop words that will be filtered during docs
        processing.
      rank (TfidfRank): Object responsible for tf-idf ranking computation.
      index (Index): Object responsible for data indexing.

    """    

    def __init__(self, tweets, stop_words):
        self.tweets         = tweets
        self.stop_words     = stop_words
        self.rank           = pickle.load(open(CUR_DIR + "/../data/rank.p", "rb"))
        self.index          = pickle.load(open(CUR_DIR + "/../data/index.p", "rb"))
        self.search_results = []

    def search(self, query, n_results=10):
        """Return indexed documents given a query of terms.

        Assumptions:
          1) We assume all terms in the provided query have to be found.
          Otherwise, an empty list will be returned. It is a simple
          assumption that can be easily changed to consider any term.

          2) We do not use positional information of the query term. It is
          not difficult whatsoever to take it into account, but it was just a
          design choice since this requirement was not specified.

        Args:
          query (str): String containing one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List of search results including the indexed
            object and its respective tf-idf score.

        """
        self.search_results = []
        terms = query.lower().split()

        # strip off stop_words terms.
        s_terms = []
        for term in terms:
            if term not in self.stop_words:
                s_terms.append(term)
        terms = s_terms

        docs_indices = self.index.search_terms(terms)

        for doc_index in docs_indices:
            indexable = self.tweets[doc_index]
            doc_score = self.rank.compute_rank(doc_index, terms)

            # highlight matching terms.
            is_highlighted = False
            for term in terms:
                if is_highlighted == False:
                        indexable.highlighted_data = indexable.original_data
                        is_highlighted = True

                pattern = re.compile(term, re.IGNORECASE)
                indexable.highlighted_data = pattern.sub("\x1b[31m" + term + "\x1b[0m", indexable.highlighted_data)
                
            result = IndexableResult(doc_score, indexable)
            self.search_results.append(result)

        self.search_results.sort(key=lambda x: x.score, reverse=True)
        return self.search_results[:n_results]

    def count(self):
        """Return number of objects already in the index.

        Returns:
          int: Number of documents indexed.

        """
        return len(self.tweets)

    def search_count(self):
        return len(self.search_results)
