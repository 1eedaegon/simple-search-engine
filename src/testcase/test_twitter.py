# -*- coding: utf-8 -*-
import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CUR_DIR + "/../")

import unittest

from twitter import TwitterDataPreprocessor
from twitter import Twitter

class TwitterTests(unittest.TestCase):

    def setUp(self):
        self.twitter = Twitter(CUR_DIR + "/test_crossfit.tweets", CUR_DIR + "/test_stop_words.txt")

    def test_data_preprocess(self):
        processor = TwitterDataPreprocessor()

        p_entry1 = processor.preprocess("\"There are no office hours for champions.\"—Paul Dietzel	@CrossFitGames")
        p_entry2 = processor.preprocess("Saturday 6-21-14 - http://t.co/ZtQWUsfal1 http://t.co/jPICqL3adi	@ReebokCrossFit1")
        p_entry3 = processor.preprocess("Crossfit Named - Kristan Clever  Valley CrossFit :	@Cleverhandz")

        text1 = p_entry1[0].strip()
        screen_name1 = p_entry1[1].strip()
        text2 = p_entry2[0].strip()
        screen_name2 = p_entry2[1].strip()
        text3 = p_entry3[0].strip()
        screen_name3 = p_entry3[1].strip()

        self.assertEqual(text1, "there are no office hours for champions paul dietzel")
        self.assertEqual(screen_name1, "crossfitgames")
        self.assertEqual(text2, "saturday 6 21 14 http t co ztqwusfal1 http t co jpicql3adi")
        self.assertEqual(screen_name2, "reebokcrossfit1")
        self.assertEqual(text3, "crossfit named kristan clever valley crossfit")
        self.assertEqual(screen_name3, "cleverhandz")

    def test_twitter_data_building(self):
        self.twitter.load_tweets_and_build_index()


if __name__ == '__main__':
    unittest.main()
