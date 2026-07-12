import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import json
import numpy as np

# Adjust sys.path or import from cli directly
from cli.evaluation_cli import main

class TestEvaluationCLI(unittest.TestCase):
    @patch('cli.evaluation_cli.open')
    @patch('cli.evaluation_cli.json.load')
    @patch('cli.evaluation_cli.InvertedIndex')
    @patch('cli.evaluation_cli.SemanticSearch')
    @patch('sys.argv')
    def test_evaluation_flow(self, mock_argv, mock_semantic_search_cls, mock_inverted_index_cls, mock_json_load, mock_open):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda s: {
            slice(None, None, None): ["evaluation_cli.py", "--limit", "2", "--search-method", "hybrid"],
        }[s]
        mock_argv.copy.return_value = ["evaluation_cli.py", "--limit", "2", "--search-method", "hybrid"]
        
        # We can also mock argparse parsing directly or mock argv list
        # Mock argv list
        import sys
        sys.argv = ["evaluation_cli.py", "--limit", "2", "--search-method", "hybrid"]

        # Mock dataset JSON load
        mock_json_load.return_value = {
            "test_cases": [
                {
                    "query": "test query",
                    "relevant_docs": ["Movie A", "Movie B"]
                }
            ]
        }

        # Mock InvertedIndex
        mock_index_instance = MagicMock()
        mock_index_instance.doc_map = {
            "1": "Movie A\nDescription A",
            "2": "Movie B\nDescription B",
            "3": "Movie C\nDescription C"
        }
        mock_index_instance.get_documents.side_effect = lambda token: ["1", "2", "3"]
        mock_index_instance.bm25.side_effect = lambda doc_id, query: {
            "1": 3.0,
            "2": 2.0,
            "3": 1.0
        }[doc_id]
        mock_inverted_index_cls.return_value = mock_index_instance

        # Mock SemanticSearch
        mock_semantic_instance = MagicMock()
        mock_semantic_instance.embeddings = np.random.rand(3, 384)
        mock_semantic_instance.document_map = {
            0: "1",
            1: "2",
            2: "3"
        }
        mock_semantic_instance.generate_embedding.return_value = np.random.rand(384)
        mock_semantic_search_cls.return_value = mock_semantic_instance

        # Capture print output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with patch('os.path.exists', return_value=True):
                main()
                
            output = mock_stdout.getvalue()
            
            # Assertions on standard output format
            self.assertIn("k=2", output)
            self.assertIn("- Query: test query", output)
            self.assertIn("Precision@2:", output)
            self.assertIn("Retrieved:", output)
            self.assertIn("Relevant: Movie A, Movie B", output)

if __name__ == '__main__':
    unittest.main()
