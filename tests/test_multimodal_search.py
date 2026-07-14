import os
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

try:
    from cli.multimodal_search_cli import main
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../cli")))
    from multimodal_search_cli import main

class TestMultimodalSearch(unittest.TestCase):
    @patch("cli.lib.multimodal_search.MultimodalSearch")
    @patch("sys.argv")
    def test_verify_image_embedding_flow(self, mock_argv, mock_multimodal_search_class):
        # Configure arguments
        mock_argv.__getitem__.side_effect = lambda x: ["cli/multimodal_search_cli.py", "verify_image_embedding", "test_image.jpg"][x]
        mock_argv.__len__.return_value = 3

        # Mock MultimodalSearch
        mock_instance = MagicMock()
        import numpy as np
        mock_instance.embed_image.return_value = np.zeros(512)
        mock_multimodal_search_class.return_value = mock_instance

        # Capture output
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            main()

        output_str = captured_output.getvalue()
        
        # Verify output
        self.assertIn("Embedding shape: 512 dimensions", output_str)
        mock_instance.embed_image.assert_called_once_with("test_image.jpg")

if __name__ == "__main__":
    unittest.main()
