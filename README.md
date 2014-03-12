# Simple Search Engine

### Definitions


The columns are `id`, `title`, `author`, so in this case:
`id -> 800 title -> The plays author -> Oscar Wilde`

### Sample Output

```
score: 32.6460707315, id: 570698, title: Greuze, author: Alys Eyre Macklin
score: 32.6460707315, id: 39325, title: Greuze, author: Alys Eyre Macklin
score: 19.9661713056, id: 642628, title: Twenty-nine tales from the French, author: Alys Eyre Macklin
score: 12.2744343179, id: 723442, title: Captain Macklin : his memoirs, author: Richard Harding Davis,Walter Appleton Clark
Enter a query, or hit enter to quit
```

### Code

### Solution

The proposed problems were solved using Python `v2.7.5` the following libraries:

- numpy `v1.6.2`: (matrices operations and other utilities)
- scipy `v0.11.0` (sparse matrix data structures)

The current implementation follows [Google Style Python]
(http://google-styleguide.googlecode.com/svn/trunk/pyguide.html).

#### Contents:
 - `src/search.py`: Module containing search implementation
 - `src/test_search.py`: Module containing search unit tests
 - `src/book.py`: Module containing search abstraction for the context of books
 - `src/test_book.py`: Module containing books search unit tests
 - `solution.py`: Command line interface for books search

#### Running the application
    $ python solution.py --data "./data/title_author.tab.txt"

#### Running the unit tests
    $ python src/test_search.py
    $ python src/test_book.py

#### Comments
The current implementation proposes a general framework for indexing and ranking documents. The classes `SearchEngine`, `Index`, `TfidfRank`, `Indexable` and `IndexableResult` are not limited to the context of books and can be used in other applications.

A simple benchmark was performed to evaluate some results:

##### Parameters
- Machine: Intel i5 dual-core 2.5GHz 8GB RAM
- Dataset: 1284904 documents and 230885 terms in vocabulary

##### Results
- Index building time: 406.88 sec (6.8 min)
- Memory usage: around 3.0GB
- Average time of query search: 0.5 sec

##### Console output
```
2014-03-12 13:18:21,096 INFO [Main] Loading books...
2014-03-12 13:18:21,096 INFO [Book] Loading books from file...
2014-03-12 13:19:07,219 INFO [Search] Start search engine (Indexing | Ranking)...
2014-03-12 13:19:15,145 INFO [Ranking] Vocabulary assembled with terms count 230885
2014-03-12 13:19:15,145 INFO [Ranking] Starting tf computation...
2014-03-12 13:20:15,647 INFO [Ranking] Starting tf-idf computation...
2014-03-12 13:20:16,429 INFO [Ranking] Starting tf-idf norm computation...
2014-03-12 13:20:16,429 INFO [Index] Building index...
2014-03-12 13:20:16,429 INFO [Benchmark] Function = load_books, Time = 406.88 sec
2014-03-12 13:20:16,429 INFO [Main] Done loading books, 1284904 docs in index
Enter a query, or hit enter to quit:
```