import os
import unittest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO

try:
    from cli.augmented_generation_cli import main
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../cli")))
    from augmented_generation_cli import main

class TestAugmentedGeneration(unittest.TestCase):
    def setUp(self):
        self.movies = [
            {"id": "1", "title": "Jurassic Park", "description": "Cloned dinosaurs go wild in a park."},
            {"id": "2", "title": "Carnosaur", "description": "Dinosaur genetically engineered in a lab."},
            {"id": "3", "title": "A Sound of Thunder", "description": "Time travelers hunt dinosaurs."},
            {"id": "4", "title": "The Lost World", "description": "Sequel to Jurassic Park."},
            {"id": "5", "title": "We're Back! A Dinosaur's Story", "description": "Smart dinosaurs in NYC."}
        ]
        
    @patch("cli.augmented_generation_cli.OpenAI")
    @patch("cli.augmented_generation_cli.SemanticSearch")
    @patch("cli.augmented_generation_cli.InvertedIndex")
    @patch("builtins.open")
    @patch("cli.augmented_generation_cli.load_dotenv")
    @patch("os.environ.get")
    @patch("sys.argv")
    def test_rag_command_flow(self, mock_argv, mock_env_get, mock_load_dotenv, mock_open, mock_inverted_index_class, mock_semantic_search_class, mock_openai):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda x: ["cli/augmented_generation_cli.py", "rag", "dinosaur"][x]
        mock_argv.__len__.return_value = 3

        # Mock env vars
        mock_env_get.side_effect = lambda key, default=None: {
            "OPENROUTER_API_KEY": "fake_key",
            "HF_ACCESS_TOKEN": None,
            "HF_TOKEN": None,
        }.get(key, default)

        # Mock movies dataset load
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = json.dumps({"movies": self.movies})
        mock_open.return_value = mock_file

        # Mock InvertedIndex
        mock_idx = MagicMock()
        mock_idx.doc_map = {"5": "We're Back! A Dinosaur's Story", "1": "Jurassic Park", "4": "The Lost World", "2": "Carnosaur", "3": "A Sound of Thunder"}
        mock_idx.get_documents.side_effect = lambda token: ["5", "1", "4", "2", "3"]
        mock_idx.bm25.side_effect = lambda doc_id, query: {"5": 5.0, "1": 4.0, "4": 3.0, "2": 2.0, "3": 1.0}[doc_id]
        mock_inverted_index_class.return_value = mock_idx

        # Mock SemanticSearch
        mock_sem = MagicMock()
        mock_sem.document_map = {0: "5", 1: "1", 2: "4", 3: "2", 4: "3"}
        mock_sem.embeddings = np.array([[1.0], [0.8], [0.6], [0.4], [0.2]])
        mock_sem.generate_embedding.return_value = np.array([1.0])
        mock_semantic_search_class.return_value = mock_sem

        # Mock OpenAI chat completion
        mock_openai_instance = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "Dinosaurs are awesome."
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_chat.completions.create.return_value = mock_completion
        mock_openai_instance.chat = mock_chat
        mock_openai.return_value = mock_openai_instance

        # Capture output
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            main()

        output_str = captured_output.getvalue()
        
        # Verify stdout format
        self.assertIn("Search Results:", output_str)
        self.assertIn("- We're Back! A Dinosaur's Story", output_str)
        self.assertIn("- Jurassic Park", output_str)
        self.assertIn("- The Lost World", output_str)
        self.assertIn("- Carnosaur", output_str)
        self.assertIn("- A Sound of Thunder", output_str)
        self.assertIn("RAG Response:", output_str)
        self.assertIn("Dinosaurs are awesome.", output_str)

        # Verify correct prompt structure
        args, kwargs = mock_chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Retrieved Movies:", messages[1]["content"])
        self.assertIn("Jurassic Park", messages[1]["content"])

if __name__ == "__main__":
    unittest.main()
