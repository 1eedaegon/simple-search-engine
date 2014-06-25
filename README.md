# Simple Search Engine

### Definitions
- `tf*idf` – term frequency times inverse document frequency – read more [here](http://en.wikipedia.org/wiki/Tf%E2%80%93idf);
- `query` - a search query which can have multiple words / terms;
- `term` – a single word in a query.

### Goal
Given a set of tweet and screen name data from several tweets, implement a `tf*idf` index in memory which does the following:
- Read English text data into the index;
- For a given `query`, output the top 10 results ranked by their `tf*idf` scores.

### Data
`athletes.tweets` – This file contains athletes' tweets multiple lines taken from the tab-delimited tweet, screen_name: `There are no office hours for champions. —Paul Dietzel	@CrossFitGames`

The columns are `tweet`, `screen_name`, so in this case:
`tweet → There are no office hours for champions. —Paul Dietzel screen_name → @CrossFitGames`

### Sample Output
Your solution should output something similar to the following, but does not need to be exactly the same:

```
$ python index.py
2014-06-23 13:06:14,324 INFO [Main] Initializing ...
2014-06-23 13:06:20,586 INFO [Twitter] Function = load_tweets, Elapsed Time = 6.26 sec
2014-06-23 13:06:21,855 INFO [Ranker] Vocabulary assembled with terms count 122,466, docs count 79,331
2014-06-23 13:06:21,856 INFO [Ranker] Starting tf computation ...
2014-06-23 13:06:31,394 INFO [Ranker] Starting tf-idf computation ...
2014-06-23 13:06:37,138 INFO [Twitter] Function = load_tweets_and_build_index, Elapsed Time = 22.81 sec
2014-06-23 13:06:37,138 INFO [Main] Initialized. 79,331 docs indexed.

$ python search.py
2014-06-23 14:49:57,567 INFO [Main] Initializing ...
2014-06-23 14:50:03,804 INFO [Twitter] Function = load_tweets, Elapsed Time = 6.24 sec
2014-06-23 14:50:08,681 INFO [Twitter] Function = load_tweets_and_load_index, Elapsed Time = 11.11 sec
2014-06-23 14:50:08,681 INFO [Main] Initialized. 79,331 docs loaded.
Enter a query, or enter 'quit' to quit: crossfit
2014-06-23 14:50:11,777 INFO [Twitter] Function = search_tweets, Elapsed Time = 0.03 sec
1,273 results.
```
![sample output](http://media.tumblr.com/a3a131c37799185bd5a6cd1d4509679c/tumblr_inline_n7nje9JNFH1qzgoac.png)

### Code
You may write your solution in either Python, Java, C#, C, or C++. If you need to make any assumptions in your code, clearly document them in the comments. On startup, your code should read the given data file, then prompt the user for queries in a loop (reading from stdin), outputting the search results in a reasonable text format.

### Solution

The proposed problems were solved using Python `v2.7.5` the following libraries:

- numpy `v1.8`: (matrices operations and other utilities)
- scipy `v0.13.3` (sparse matrix data structures)

The current implementation follows [Google Style Python]
(http://google-styleguide.googlecode.com/svn/trunk/pyguide.html).

#### Contents:
 - `src/indexer.py`: Module containing index implementation
 - `src/searcher.py`: Module containing search implementation
 - `src/ranker.py`: Module containing rank implementation
 - `src/twitter.py`: Module containing search abstraction for the context of tweets
 - `index.py`: Command line interface for tweets index
 - `search.py`: Command line interface for tweets search

#### Running the application
    $ python index.py
    $ python search.py

#### Comments
The current implementation proposes a general framework for indexing and ranking documents. The classes `Searcher`, `Indexer`, `Index`, `Rank`, `Indexable` and `IndexableResult` are not limited to the context of tweets and can be used in other applications.

A simple benchmark was performed to evaluate some results:

##### Parameters
- Machine: Macbook Pro 15" 2.3 GHz Intel Core i7
- Dataset: 100,000 documents and 148,843 terms in vocabulary

##### Results
- Index building time: 28.51 sec
- Memory usage: around 362 MB
- File size:
 - athletes.tweets: 11.1 MB
 - index.p: 13.9 MB
 - rank.p: 84.1 MB
- Query time: depends on query results. around 0.01 ~ 0.8 sec
