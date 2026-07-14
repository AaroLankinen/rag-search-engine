import os
import unittest
import json
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
    @patch("cli.augmented_generation_cli.HybridSearch")
    @patch("builtins.open")
    @patch("cli.augmented_generation_cli.load_dotenv")
    @patch("os.environ.get")
    @patch("sys.argv")
    def test_rag_command_flow(self, mock_argv, mock_env_get, mock_load_dotenv, mock_open, mock_hybrid_search_class, mock_openai):
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

        # Mock HybridSearch
        mock_hybrid_search_instance = MagicMock()
        mock_hybrid_search_instance.rrf_search.return_value = [
            {"document": self.movies[4]},
            {"document": self.movies[0]},
            {"document": self.movies[3]},
            {"document": self.movies[1]},
            {"document": self.movies[2]}
        ]
        mock_hybrid_search_class.return_value = mock_hybrid_search_instance

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

    @patch("cli.augmented_generation_cli.OpenAI")
    @patch("cli.augmented_generation_cli.HybridSearch")
    @patch("builtins.open")
    @patch("cli.augmented_generation_cli.load_dotenv")
    @patch("os.environ.get")
    @patch("sys.argv")
    def test_summarize_command_flow(self, mock_argv, mock_env_get, mock_load_dotenv, mock_open, mock_hybrid_search_class, mock_openai):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda x: ["cli/augmented_generation_cli.py", "summarize", "dinosaur"][x]
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

        # Mock HybridSearch
        mock_hybrid_search_instance = MagicMock()
        mock_hybrid_search_instance.rrf_search.return_value = [
            {"document": self.movies[4]},
            {"document": self.movies[0]},
            {"document": self.movies[3]},
            {"document": self.movies[1]},
            {"document": self.movies[2]}
        ]
        mock_hybrid_search_class.return_value = mock_hybrid_search_instance

        # Mock OpenAI chat completion
        mock_openai_instance = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "Summary of dinosaur movies."
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
        # Note the double space before the dash in search results
        self.assertIn("  - We're Back! A Dinosaur's Story", output_str)
        self.assertIn("  - Jurassic Park", output_str)
        self.assertIn("  - The Lost World", output_str)
        self.assertIn("  - Carnosaur", output_str)
        self.assertIn("  - A Sound of Thunder", output_str)
        self.assertIn("LLM Summary:", output_str)
        self.assertIn("Summary of dinosaur movies.", output_str)

        # Verify correct prompt structure
        args, kwargs = mock_chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Movies to summarize:", messages[1]["content"])
        self.assertIn("Jurassic Park", messages[1]["content"])

    @patch("cli.augmented_generation_cli.OpenAI")
    @patch("cli.augmented_generation_cli.HybridSearch")
    @patch("builtins.open")
    @patch("cli.augmented_generation_cli.load_dotenv")
    @patch("os.environ.get")
    @patch("sys.argv")
    def test_citations_command_flow(self, mock_argv, mock_env_get, mock_load_dotenv, mock_open, mock_hybrid_search_class, mock_openai):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda x: ["cli/augmented_generation_cli.py", "citations", "dinosaur"][x]
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

        # Mock HybridSearch
        mock_hybrid_search_instance = MagicMock()
        mock_hybrid_search_instance.rrf_search.return_value = [
            {"document": self.movies[4]},
            {"document": self.movies[0]},
            {"document": self.movies[3]},
            {"document": self.movies[1]},
            {"document": self.movies[2]}
        ]
        mock_hybrid_search_class.return_value = mock_hybrid_search_instance

        # Mock OpenAI chat completion
        mock_openai_instance = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "Answer with citations: [1] and [2]."
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
        # Note the double space before the dash in search results
        self.assertIn("  - We're Back! A Dinosaur's Story", output_str)
        self.assertIn("  - Jurassic Park", output_str)
        self.assertIn("  - The Lost World", output_str)
        self.assertIn("  - Carnosaur", output_str)
        self.assertIn("  - A Sound of Thunder", output_str)
        self.assertIn("LLM Answer:", output_str)
        self.assertIn("Answer with citations: [1] and [2].", output_str)

        # Verify correct prompt structure
        args, kwargs = mock_chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Answer the query below and give information based on the provided documents.", messages[0]["content"])
        self.assertIn("[1] Title: We're Back! A Dinosaur's Story", messages[0]["content"])
        self.assertIn("[2] Title: Jurassic Park", messages[0]["content"])

    @patch("cli.augmented_generation_cli.OpenAI")
    @patch("cli.augmented_generation_cli.HybridSearch")
    @patch("builtins.open")
    @patch("cli.augmented_generation_cli.load_dotenv")
    @patch("os.environ.get")
    @patch("sys.argv")
    def test_question_command_flow(self, mock_argv, mock_env_get, mock_load_dotenv, mock_open, mock_hybrid_search_class, mock_openai):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda x: ["cli/augmented_generation_cli.py", "question", "dinosaur?"][x]
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

        # Mock HybridSearch
        mock_hybrid_search_instance = MagicMock()
        mock_hybrid_search_instance.rrf_search.return_value = [
            {"document": self.movies[0]},
            {"document": self.movies[1]},
            {"document": self.movies[2]},
            {"document": self.movies[3]},
            {"document": self.movies[4]}
        ]
        mock_hybrid_search_class.return_value = mock_hybrid_search_instance

        # Mock OpenAI chat completion
        mock_openai_instance = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "This is the answer."
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
        self.assertIn("  - Jurassic Park", output_str)
        self.assertIn("Answer:", output_str)
        self.assertIn("This is the answer.", output_str)

        # Verify correct prompt structure
        args, kwargs = mock_chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Question: dinosaur?", messages[1]["content"])
        self.assertIn("Jurassic Park", messages[1]["content"])

if __name__ == "__main__":
    unittest.main()
