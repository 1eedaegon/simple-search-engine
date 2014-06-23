#!/usr/bin/python
import os
import logging

from src import twitter

# Log initialization
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

CUR_DIR             = os.path.dirname(os.path.realpath(__file__))
CATALOG_FILENAME    = CUR_DIR + "/data/athletes.tweets"
STOP_WORDS_FILENAME = CUR_DIR + "/data/stop_words.txt"
QUERY_INPUT_MESSAGE = "Enter a query, or enter 'quit' to quit: "

def execute_search():
    """Capture query from STDIN and display the result on STDOUT.

    The query of terms is executed against an indexed data structure
    containing tweets' information.

    """
    tw = twitter.Twitter(CATALOG_FILENAME, STOP_WORDS_FILENAME)    
    logging.info("[Main] Initializing ...")
    tw.load_tweets_and_load_index()
    logging.info("[Main] Initialized. %s docs loaded.", "{:,}".format(tw.tweets_count()))

    query = None
    while True:
        query = raw_input(QUERY_INPUT_MESSAGE)
        if query == '':
            continue
        if query == 'quit':
            break

        search_results = tw.search_tweets(query)
        print search_results


if __name__ == '__main__':
    execute_search()
