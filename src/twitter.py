# -*- coding: utf-8 -*-
import os
import time
import re
import unicodedata
import logging
import pickle

from indexer import Indexable
from indexer import Indexer

from searcher import Searcher

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def timed(fn):
    """Decorator used to benchmark functions runtime.

    """

    def wrapped(*arg, **kw):
        ts = time.time()
        result = fn(*arg, **kw)
        te = time.time()

        logging.info("[Twitter] Function = %s, Elapsed Time = %2.2f sec" \
              % (fn.__name__, (te - ts)))

        return result

    return wrapped


class Tweet(Indexable):
    """Class encapsulating a specific behavior of indexed books.

    Args:
      docid (int): Identifier of indexable objects.
      indexable_data (str): Processed data to be indexed.
      original_data (str): Original data
      highlighted_data (str): Temporarily highlighted data during search

    """

    def __init__(self, docid, indexable_data, original_data, highlighted_data = None):
        Indexable.__init__(self, docid, indexable_data)
        self.original_data = original_data
        self.highlighted_data = highlighted_data

    def __repr__(self):
        return "docid: %s, tweet: %s" % \
               (self.docid, self.highlighted_data)


class Twitter(object):
    """Class representing a inventory of books.

    Args:
      tweets_filename (str): File name containing tweets.
      stop_words_filename (str): File name containing stop words.

    Attributes:
      teets(list): List of original tweets.
      stop_words (list): List of stop words.
      indexer (Indexer): Object responsible for indexing tweets.
      searcher (Searcher): Object responsible for searching tweets.

    """

    _TWEET_META_TEXT_INDEX = 0
    _TWEET_META_SCREEN_NAME_INDEX = 1

    _NO_RESULTS_MESSAGE = "Sorry, no results."

    def __init__(self, tweets_filename, stop_words_filename):
        self.tweets             = []
        self.tweets_filename    = tweets_filename
        self.stop_words         = self.__load_stop_words(stop_words_filename)
        self.indexer            = Indexer(self.stop_words)
        self.searcher           = []

    @timed
    def load_tweets(self):
        """Load tweets from a file name.

        This method leverages the iterable behavior of File objects
        that automatically uses buffered IO and memory management handling
        effectively large files.

        """
        docid = 0
        processor = TwitterDataPreprocessor()
        with open(self.tweets_filename) as catalog:
            for entry in catalog:
                # preprocessing
                p_entry = processor.preprocess(entry)

                text = p_entry[self._TWEET_META_TEXT_INDEX].strip()
                screen_name = ''
                if len(p_entry) > 1:
                    screen_name = p_entry[self._TWEET_META_SCREEN_NAME_INDEX].strip()
                
                indexable_data = text + ' ' + screen_name
                original_data = entry

                tweet = Tweet(docid, indexable_data, original_data)
                self.tweets.append(tweet)
                docid += 1

    @timed
    def load_tweets_and_build_index(self):
        """Load tweets from a file name, build index, compute ranking and save them all.

        """
        self.load_tweets()
        self.indexer.build_and_save(self.tweets)

    @timed
    def load_tweets_and_load_index(self):
        """Load tweets from a file name and load index from a file name.

        """
        self.load_tweets()
        self.searcher = Searcher(self.tweets, self.stop_words)

    @timed
    def search_tweets(self, query, n_results=10):
        """Search tweets according to provided query of terms.

        The query is executed against the indexed tweets, and a list of tweets
        compatible with the provided terms is return along with their tf-idf
        score.

        Args:
          query (str): Query string with one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List containing tweets and their respective
            tf-idf scores.

        """
        result = ''
        if len(query) > 0:
            result = self.searcher.search(query, n_results)

        if len(result) > 0:
            return "{:,}".format(self.searcher.search_count()) \
                + " results.\n\n" \
                + "".join([str(indexable) for indexable in result])
        return self._NO_RESULTS_MESSAGE        

    def tweets_count(self):
        """Return number of loaded tweets.

        Returns:
          int: Number of loaded tweets.

        """
        return len(self.tweets)
    
    def __load_stop_words(self, stop_words_filename):
        """Load stop words that will be filtered during docs processing.

        Stop words are words which are filtered out prior to
        processing of natural language data. There is not one definite
        list of stop words but we are using the list in `stop_words.txt` file.

        Returns:
          list str: List of English stop words.

        """
        stop_words = {}
        with open(stop_words_filename) as stop_words_file:
            for word in stop_words_file:
                stop_words[word.strip()] = True
        return stop_words

class TwitterDataPreprocessor(object):
    """Preprocessor for tweet entries.

    """

    _EXTRA_SPACE_REGEX = re.compile(r'\s+', re.IGNORECASE)
    _SPECIAL_CHAR_REGEX = re.compile(
        # detect punctuation characters
        r'(?P<p>(\.+)|(\?+)|(!+)|(:+)|(;+)|'
        # detect special characters
        r'(\(+)|(\)+)|(\}+)|(\{+)|("+)|(-+)|(\[+)|(\]+)|(\@+)|(\/+)|(\#+)|'
        # detect commas NOT between numbers
        r'(?<!\d)(,+)(?!=\d)|(\$+))')

    def preprocess(self, entry):
        """Preprocess an entry to a sanitized format.

        The preprocess steps applied to the book entry is the following::
          1) All non-accents are removed;
          2) Special characters are replaced by whitespaces (i.e. -, [, etc.);
          3) Punctuation marks are removed;
          4) Additional whitespaces between replaced by only one whitespaces.

        Args:
          entry (str): Book entry in string format to be preprocess.

        Returns:
          str: Sanitized book entry.

        """
        f_entry = entry.lower()
        f_entry = f_entry.replace('\t', '||').strip()
        f_entry = self.strip_accents(unicode(f_entry, "utf-8"))
        f_entry = self._SPECIAL_CHAR_REGEX.sub(' ', f_entry)
        f_entry = self._EXTRA_SPACE_REGEX.sub(' ', f_entry)

        p_entry = f_entry.split('||')

        return p_entry

    def strip_accents(self, text):
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
