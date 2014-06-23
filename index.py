#!/usr/bin/python
import os
import logging
from src import twitter

# Log initialization
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

CUR_DIR             = os.path.dirname(os.path.realpath(__file__))
CATALOG_FILENAME    = CUR_DIR + "/data/athletes.tweets"
STOP_WORDS_FILENAME = CUR_DIR + "/data/stop_words.txt"

def execute_index():
    """Create an index of quality from tf-idf variables to enable rank ordering.

    """
    query = None
    tw = twitter.Twitter(CATALOG_FILENAME, STOP_WORDS_FILENAME)
    logging.info("[Main] Initializing ...")
    tw.load_tweets_and_build_index()
    docs_number = tw.tweets_count()
    logging.info("[Main] Initialized. %s docs indexed.", "{:,}".format(docs_number))


if __name__ == '__main__':
    execute_index()
