# -*- coding: utf-8 -*-
import os
import time
import logging
import pickle

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

        logging.info("[Indexer] Function = %s, Elapsed Time = %2.2f sec" \
              % (fn.__name__, (te - ts)))

        return result

    return wrapped

class Indexable(object):
    """Class representing an object that can be indexed.

    It is a general abstraction for indexable objects and can be used in
    different contexts.

    Args:
      docid (int): Identifier of indexable objects.
      data (str): Plain text with data to be indexed.

    Attributes:
      docid (int): Identifier of indexable objects.
      words_count (dict): Dictionary containing the unique words from
        `data` and their frequency.

    """

    def __init__(self, docid, indexable_data):
        self.docid = docid
        self.words_count = defaultdict(int)

        for word in indexable_data.split():
            self.words_count[word] += 1

    # def __repr__(self):
    #     return " ".join(self.words_count.keys()[:10])

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def words_generator(self, stop_words):
        """Yield unique words extracted from indexed metadata.

        Args:
          stop_words (list of str): List with words that must be filtered.

        Yields:
          str: Unique words from indexed metadata.

        """
        for word in self.words_count.keys():
            if word not in stop_words:
                yield word

    def count_for_word(self, word):
        """Frequency of a given word from indexed metadata.

        Args:
          word (str): Word whose the frequency will be retrieved.

        Returns:
          int: Number of occurrences of a given word.

        """
        return self.words_count[word] if word in self.words_count else 0

class Index(object):
    """Class responsible for indexing objects.

    Note:
      In case of a indexer object, we dropped the runtime complexity for a
      search by increasing the space complexity. It is the traditional
      trade-off and here we are more interested in a lightening fast
      search then saving some space. This logic may have to be revisited if the
      index become too large.

    Args:
      term_index (dict): Dictionary containing a term as key and a list of all
        the documents that contain that key/term as values
      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    Attributes:
      term_index (dict): Dictionary containing a term as key and a list of all
        the documents that contain that key/term as values
      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    """

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.term_index = defaultdict(list)

    def build_index(self, tweets):
        """Build index the given indexable tweets.

        Args:
          tweets (list of Indexable): Indexed tweets that will be
            considered during search.

        """
        for position, indexable in enumerate(tweets):
          for word in indexable.words_generator(self.stop_words):
              # build dictionary where term is the key and an array
              # of the IDs of indexable object containing the term
              self.term_index[word].append(position)

    def search_terms(self, terms):
        """Search for terms in indexed documents.

        Args:
          terms (list of str): List of terms considered during the search.

        Returns:
          list of int: List containing the index of indexed objects that
            contains the query terms.

        """
        docs_indices = []
        is_first_term = True
        for term_index, term in enumerate(terms):

            # ignore stop_words query
            if term in self.stop_words:
                continue

            # keep only docs that contains all terms
            if term not in self.term_index:
                docs_indices = []
                break

            # compute intersection between results
            # there is room for improvements in this part of the code
            docs_with_term = self.term_index[term]
            if is_first_term == True:
                docs_indices = docs_with_term
                is_first_term = False
            else:
                # if there are two and more terms results then AND list of lists
                docs_indices = set(docs_indices) & set(docs_with_term)

        return list(docs_indices)


class Indexer(object):

    def __init__(self, stop_words):
        self.rank = Rank(stop_words)
        self.index = Index(stop_words)

    def build_and_save(self, tweets):
        """Start indexing and ranking initialization.

        The current implementation initialize the ranking and indexing of
        added objects. The code below is not very efficient as it iterates over
        all indexed objects twice, but can be improved easily with generators.

        """
        self.index.build_index(tweets)
        self.rank.build_rank(tweets)

        pickle.dump(self.index, open(CUR_DIR + "/../data/index.p", "wb"))
        pickle.dump(self.rank,  open(CUR_DIR + "/../data/rank.p", "wb"))