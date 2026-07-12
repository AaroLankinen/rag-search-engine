import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import json
import sys
import os

from cli.hybrid_search_cli import main

class TestHybridSearchEvaluate(unittest.TestCase):
    @patch('cli.hybrid_search_cli.argparse.ArgumentParser.parse_args')
    @patch('cli.hybrid_search_cli.open', new_callable=lambda: MagicMock(return_value=StringIO('{"movies": [{"id": 1, "title": "Ted", "description": "A talking teddy bear."}, {"id": 2, "title": "Paddington", "description": "A polite bear from Peru."}]}')))
    @patch('cli.lib.hybrid_search.HybridSearch')
    @patch('openai.OpenAI')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fake-openrouter-key'})
    def test_rrf_search_evaluate(self, mock_openai_cls, mock_hybrid_search_cli, mock_open, mock_parse_args):

        # Mock parsed arguments for: rrf-search "teddy bear" --evaluate --limit 2
        mock_args = MagicMock()
        mock_args.command = "rrf-search"
        mock_args.query = "teddy bear"
        mock_args.k = 60
        mock_args.limit = 2
        mock_args.data_file = "data/movies.json"
        mock_args.save_dir = "cache"
        mock_args.enhance = None
        mock_args.rerank_method = None
        mock_args.debug = False
        mock_args.evaluate = True
        mock_parse_args.return_value = mock_args

        # Mock hybrid search results
        mock_hybrid_search_instance = MagicMock()
        mock_hybrid_search_instance.rrf_search.return_value = [
            {"document": {"id": 1, "title": "Ted", "description": "A talking teddy bear."}, "rrf_score": 0.03, "bm25_score": 1.0, "semantic_score": 1.0},
            {"document": {"id": 2, "title": "Paddington", "description": "A polite bear from Peru."}, "rrf_score": 0.02, "bm25_score": 0.8, "semantic_score": 0.8}
        ]
        mock_hybrid_search_cli.return_value = mock_hybrid_search_instance

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        
        # LLM returns JSON score mapping
        mock_message.content = '{"1": 3, "2": 2}'
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        # Capture output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()

        # Assert output format is correct
        self.assertIn("1. Ted: 3/3", output)
        self.assertIn("2. Paddington: 2/3", output)

if __name__ == '__main__':
    unittest.main()
