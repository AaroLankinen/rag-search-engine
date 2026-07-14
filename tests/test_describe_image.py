import os
import unittest
import json
from unittest.mock import patch, MagicMock
from io import StringIO

try:
    from cli.describe_image_cli import main
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../cli")))
    from describe_image_cli import main

class TestDescribeImage(unittest.TestCase):
    @patch("cli.describe_image_cli.OpenAI")
    @patch("builtins.open")
    @patch("cli.describe_image_cli.mimetypes.guess_type")
    @patch("cli.describe_image_cli.load_dotenv")
    @patch("os.environ.get")
    @patch("sys.argv")
    def test_describe_image_cli_flow(self, mock_argv, mock_env_get, mock_load_dotenv, mock_guess_type, mock_open, mock_openai):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda x: ["cli/describe_image_cli.py", "--image", "test.png", "--query", "dinosaur movie"][x]
        mock_argv.__len__.return_value = 5

        # Mock env vars
        mock_env_get.side_effect = lambda key, default=None: {
            "OPENROUTER_API_KEY": "fake_key",
        }.get(key, default)

        # Mock mimetypes.guess_type
        mock_guess_type.return_value = ("image/png", None)

        # Mock image file load
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = b"fake_image_bytes"
        mock_open.return_value = mock_file

        # Mock OpenAI chat completion
        mock_openai_instance = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_usage = MagicMock()
        
        mock_message.content = "Rewritten dinosaur movie query"
        mock_choice.message = mock_message
        mock_usage.total_tokens = 150
        mock_completion.choices = [mock_choice]
        mock_completion.usage = mock_usage
        mock_chat.completions.create.return_value = mock_completion
        mock_openai_instance.chat = mock_chat
        mock_openai.return_value = mock_openai_instance

        # Capture output
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            main()

        output_str = captured_output.getvalue()
        
        # Verify stdout format
        self.assertIn("Rewritten query: Rewritten dinosaur movie query", output_str)
        self.assertIn("Total tokens:    150", output_str)

        # Verify correct prompt structure
        args, kwargs = mock_chat.completions.create.call_args
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Given the included image and text query", messages[0]["content"])
        self.assertEqual(messages[1]["content"][0]["text"], "dinosaur movie")
        self.assertEqual(messages[1]["content"][1]["image_url"]["url"], "data:image/png;base64,ZmFrZV9pbWFnZV9ieXRlcw==")

if __name__ == "__main__":
    unittest.main()
