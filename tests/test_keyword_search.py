import unittest
import os
import tempfile
import json
import math
import collections
from io import StringIO
from unittest.mock import patch, MagicMock

# Import the module components.
# Adding workspace root to path if needed is handled by python -m unittest.
from cli.keyword_search_cli import (
    preprocess_text,
    tokenize_term,
    load_stopwords,
    InvertedIndex,
    build_command,
    main
)
from cli.constants import K1, B

class TestTextPreprocessing(unittest.TestCase):
    def test_preprocess_text_standard(self):
        text = "Running fast, dogs are barking!"
        tokens = preprocess_text(text)
        # "running" stems to "run", "fast" to "fast", "dogs" to "dog", "are" to "are", "barking" to "bark"
        # Punctuation should be removed.
        expected = ["run", "fast", "dog", "are", "bark"]
        self.assertEqual(tokens, expected)

    def test_preprocess_text_empty(self):
        self.assertEqual(preprocess_text(""), [])
        self.assertEqual(preprocess_text("!!! ???"), [])

    def test_tokenize_term(self):
        self.assertEqual(tokenize_term("Running"), "run")
        self.assertEqual(tokenize_term(""), "")
        self.assertEqual(tokenize_term("!!!"), "")


class TestStopwordsLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_stopwords_exists(self):
        filepath = os.path.join(self.temp_dir.name, "stopwords.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("the\nand\nrunning\n")
        
        stopwords = load_stopwords(filepath)
        # Note that the words are preprocessed, so "running" becomes "run"
        expected = {"the", "and", "run"}
        self.assertEqual(stopwords, expected)

    def test_load_stopwords_missing(self):
        filepath = os.path.join(self.temp_dir.name, "nonexistent.txt")
        # Should not raise exception, but output warning to stderr and return empty set
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            stopwords = load_stopwords(filepath)
            self.assertEqual(stopwords, set())
            self.assertIn("Warning: Data file", mock_stderr.getvalue())


class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        self.docs_list = [
            "The quick brown fox jumps over the lazy dog",
            "Dogs are running and jumping in the park",
            "Foxes love running fast"
        ]
        self.docs_dict = {
            "10": "The quick brown fox jumps over the lazy dog",
            "20": "Dogs are running and jumping in the park",
            "30": "Foxes love running fast"
        }

    def test_init_with_list(self):
        index = InvertedIndex(self.docs_list)
        # Document IDs should be strings representation of list indices
        self.assertEqual(len(index.doc_map), 3)
        self.assertEqual(index.doc_map["0"], self.docs_list[0])
        self.assertEqual(index.doc_map["1"], self.docs_list[1])
        self.assertEqual(index.doc_map["2"], self.docs_list[2])

    def test_init_with_dict(self):
        index = InvertedIndex(self.docs_dict)
        self.assertEqual(len(index.doc_map), 3)
        self.assertEqual(index.doc_map["10"], self.docs_dict["10"])
        self.assertEqual(index.doc_map["20"], self.docs_dict["20"])
        self.assertEqual(index.doc_map["30"], self.docs_dict["30"])

    def test_term_frequency_get_tf(self):
        index = InvertedIndex(self.docs_dict)
        # "dogs" preprocessed to "dog"
        # Doc 20: "Dogs are running and jumping..." contains "Dogs" once.
        self.assertEqual(index.get_tf("20", "dog"), 1)
        # Doc 10 has "dog" once ("lazy dog")
        self.assertEqual(index.get_tf("10", "dog"), 1)
        # Doc 30 does not have "dog"
        self.assertEqual(index.get_tf("30", "dog"), 0)
        # Missing doc_id
        self.assertEqual(index.get_tf("99", "dog"), 0)

    def test_avg_doc_len(self):
        index = InvertedIndex(self.docs_dict)
        # doc 10: "The quick brown fox jumps over the lazy dog" -> 9 words
        # doc 20: "Dogs are running and jumping in the park" -> 8 words
        # doc 30: "Foxes love running fast" -> 4 words
        # average = (9 + 8 + 4) / 3 = 7.0
        self.assertEqual(index.avg_doc_len, 7.0)

    def test_avg_doc_len_empty(self):
        index = InvertedIndex({})
        self.assertEqual(index.avg_doc_len, 0.0)

    def test_get_bm25_idf(self):
        index = InvertedIndex(self.docs_dict)
        # total documents N = 3
        # term "run" (from running) appears in doc 20 and doc 30, so term_match_doc_count = 2
        # IDF = log((3 - 2 + 0.5) / (2 + 0.5) + 1) = log(1.5 / 2.5 + 1) = log(1.6)
        expected_idf = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
        self.assertAlmostEqual(index.get_bm25_idf("run"), expected_idf)

        # term "fox" (from fox, foxes) appears in doc 10 and doc 30, n = 2
        self.assertAlmostEqual(index.get_bm25_idf("fox"), expected_idf)

        # term "lazy" appears in doc 10, n = 1
        expected_lazy_idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
        self.assertAlmostEqual(index.get_bm25_idf("lazi"), expected_lazy_idf)

    def test_get_bm25_tf(self):
        index = InvertedIndex(self.docs_dict)
        # doc 20 length = 8. average length = 7.0
        # term "run" appears 1 time in doc 20. tf = 1.
        # get_bm25_tf(doc_id, term, k1, b)
        # Formula: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        tf = 1
        doc_len = 8
        avg_len = 7.0
        expected_bm25_tf = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * doc_len / avg_len))
        self.assertAlmostEqual(index.get_bm25_tf("20", "run"), expected_bm25_tf)

    def test_bm25_score(self):
        index = InvertedIndex(self.docs_dict)
        # term is preprocessed inside bm25 score calculation.
        # bm25("20", "running")
        # tokenized term = "run"
        score = index.bm25("20", "running")
        expected_score = index.get_bm25_tf("20", "run") * index.get_bm25_idf("run")
        self.assertAlmostEqual(score, expected_score)

    def test_bm25_search(self):
        index = InvertedIndex(self.docs_dict)
        # Searching for "running" -> should match doc 20 and doc 30.
        # Doc 30 length is 4. Doc 20 length is 8.
        # Since doc 30 is shorter, its BM25 term frequency component will be higher.
        # Both have tf=1 for "run". IDF is the same.
        # Thus doc 30 should rank higher than doc 20.
        results = index.bm25_search("running")
        self.assertEqual(results, ["30", "20"])

    def test_search_and_get_documents(self):
        index = InvertedIndex(self.docs_dict)
        # "fox" matches doc 10, "foxes" preprocessed matches doc 30.
        # get_documents("fox") returns docs containing stemmed "fox" -> "fox" and "foxes" stem to "fox".
        # Let's check get_documents
        self.assertEqual(set(index.get_documents("fox")), {"10", "30"})
        self.assertEqual(index.get_documents("nonexistent"), [])
        self.assertEqual(index.get_documents("!!!"), [])

        # test search (standard boolean list extension of matching documents)
        results = index.search("fox dog")
        # "fox" matches 10, 30. "dog" matches 10, 20.
        # results list contains duplicates as it just extends lists.
        self.assertIn("10", results)
        self.assertIn("20", results)
        self.assertIn("30", results)

    def test_save_and_load(self):
        index = InvertedIndex(self.docs_dict)
        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "index.pkl")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "docmap.pkl")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "term_frequencies.pkl")))

            new_index = InvertedIndex({})
            new_index.load(tmpdir)
            self.assertEqual(new_index.doc_map, index.doc_map)
            self.assertEqual(new_index.index, index.index)
            # Compare term frequencies dicts
            self.assertEqual(dict(new_index.term_frequencies), dict(index.term_frequencies))

    def test_load_nonexistent(self):
        new_index = InvertedIndex({})
        # Loading from non-existent directory should print error to stderr and exit
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                new_index.load("/nonexistent/directory/path/here")
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error: Index files not found", mock_stderr.getvalue())


class TestBuildCommand(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dict = {
            "movies": [
                {"id": 101, "title": "Movie One", "description": "A great movie"},
                {"id": 102, "title": "Movie Two", "description": "Another great action movie"}
            ]
        }
        self.data_list = [
            {"id": 201, "title": "Movie Three", "description": "Sci-fi adventure"},
            {"id": 202, "title": "Movie Four", "description": "Drama movie"}
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_command_dict(self):
        json_path = os.path.join(self.temp_dir.name, "movies_dict.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.data_dict, f)

        build_command(json_path, self.temp_dir.name)
        
        # Load and verify
        idx = InvertedIndex({})
        idx.load(self.temp_dir.name)
        self.assertEqual(len(idx.doc_map), 2)
        self.assertIn("101", idx.doc_map)
        self.assertIn("102", idx.doc_map)
        self.assertIn("movi", idx.index)

    def test_build_command_list(self):
        json_path = os.path.join(self.temp_dir.name, "movies_list.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.data_list, f)

        build_command(json_path, self.temp_dir.name)
        
        # Load and verify
        idx = InvertedIndex({})
        idx.load(self.temp_dir.name)
        self.assertEqual(len(idx.doc_map), 2)
        self.assertIn("201", idx.doc_map)
        self.assertIn("202", idx.doc_map)

    def test_build_command_missing_file(self):
        json_path = os.path.join(self.temp_dir.name, "nonexistent.json")
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                build_command(json_path, self.temp_dir.name)
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error: Data file", mock_stderr.getvalue())

    def test_build_command_invalid_json(self):
        json_path = os.path.join(self.temp_dir.name, "bad.json")
        with open(json_path, "w") as f:
            f.write("{invalid json")
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                build_command(json_path, self.temp_dir.name)
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error: Data file", mock_stderr.getvalue())
            self.assertIn("not valid JSON", mock_stderr.getvalue())


class TestCLICommands(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create a mock index for CLI to load
        self.index = InvertedIndex({
            "100": "Avatar\nAn action adventure sci-fi movie with blue people",
            "200": "The Godfather\nA mafia crime drama movie",
            "300": "The Dark Knight\nA superhero action movie about Batman"
        })
        self.index.save(self.temp_dir.name)
        
        # We also need a dummy stopwords file to avoid warning
        self.stopwords_path = "data/stopwords.txt"
        
    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('sys.argv', ['keyword_search_cli.py', 'search', 'action movie', 'cache_dir_placeholder', '2'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_search(self, mock_stdout, *mocks):
        # We need to run CLI. We'll patch args and temp dir.
        with patch('cli.keyword_search_cli.load_stopwords', return_value={"a", "an", "the", "with", "about"}):
            with patch('sys.argv', ['keyword_search_cli.py', 'search', 'action movie', self.temp_dir.name, '2']):
                main()
        
        output = mock_stdout.getvalue()
        # Query preprocessed is ["action", "movie"]
        # Both "Avatar" and "The Dark Knight" contain "action" and "movie".
        # Let's see what is printed.
        self.assertIn("Searching for: action movie", output)
        self.assertIn("Document ID: 100, Title: Avatar", output)
        self.assertIn("Document ID: 300, Title: The Dark Knight", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_tf(self, mock_stdout):
        # tf <doc_id> <term> <index_dir>
        with patch('sys.argv', ['keyword_search_cli.py', 'tf', '100', 'blue', self.temp_dir.name]):
            main()
        output = mock_stdout.getvalue()
        # Term frequency of 'blue' in document '100': 1
        self.assertIn("Term frequency of 'blue' in document '100': 1", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_idf(self, mock_stdout):
        # idf <term> <index_dir>
        with patch('sys.argv', ['keyword_search_cli.py', 'idf', 'movie', self.temp_dir.name]):
            main()
        output = mock_stdout.getvalue()
        # N = 3. term 'movie' in docs 100, 200, 300 (n=3).
        # idf = math.log((3 + 1) / (3 + 1)) = 0.0
        self.assertIn("Inverse document frequency of 'movie': 0.00", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_tfidf(self, mock_stdout):
        # tfidf <doc_id> <term> <index_dir>
        with patch('sys.argv', ['keyword_search_cli.py', 'tfidf', '100', 'blue', self.temp_dir.name]):
            main()
        output = mock_stdout.getvalue()
        # doc 100 has 'blue' tf=1. n=1. idf = log((3 + 1) / (1 + 1)) = log(2) = 0.693147
        # tfidf = 0.69
        self.assertIn("TF-IDF of 'blue' in document '100': 0.69", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_bm25idf(self, mock_stdout):
        # bm25idf <term> <index_dir>
        with patch('sys.argv', ['keyword_search_cli.py', 'bm25idf', 'blue', self.temp_dir.name]):
            main()
        output = mock_stdout.getvalue()
        # BM25 IDF of 'blue': math.log((3 - 1 + 0.5)/(1 + 0.5) + 1) = log(2.5/1.5 + 1) = log(2.666) = 0.9808
        self.assertIn("BM25 IDF of 'blue': 0.98", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_bm25tf(self, mock_stdout):
        # bm25tf <doc_id> <term> <k1> <index_dir>
        with patch('sys.argv', ['keyword_search_cli.py', 'bm25tf', '100', 'blue', '1.5', self.temp_dir.name]):
            main()
        output = mock_stdout.getvalue()
        # output should print BM25 TF
        self.assertIn("BM25 TF of 'blue' in document '100':", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_bm25search(self, mock_stdout):
        # bm25search <query> --limit <limit> <index_dir>
        with patch('sys.argv', ['keyword_search_cli.py', 'bm25search', 'blue movie', '--limit', '2', self.temp_dir.name]):
            main()
        output = mock_stdout.getvalue()
        # "Avatar" (100) contains "blue" and "movie".
        self.assertIn("1. (100) Avatar - Score:", output)

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_build(self, mock_stdout):
        # We mock build_command to ensure main delegates properly
        with patch('cli.keyword_search_cli.build_command') as mock_build:
            with patch('sys.argv', ['keyword_search_cli.py', 'build', 'dummy_data.json', 'dummy_cache']):
                main()
            mock_build.assert_called_once_with('dummy_data.json', 'dummy_cache')

    @patch('sys.stdout', new_callable=StringIO)
    def test_cli_no_command(self, mock_stdout):
        # Test CLI called without subcommand prints help
        with patch('sys.argv', ['keyword_search_cli.py']):
            main()
        output = mock_stdout.getvalue()
        self.assertIn("Keyword Search CLI", output)


if __name__ == "__main__":
    unittest.main()
