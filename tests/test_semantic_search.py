import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from cli.lib.semantic_search import SemanticSearch, cosine_similarity, verify_model, embed_text, embed_query

class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for cache testing
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Mock SentenceTransformer
        self.mock_encoder_patch = patch('cli.lib.semantic_search.SentenceTransformer')
        self.mock_encoder = self.mock_encoder_patch.start()
        
        # Configure encode return value
        def mock_encode(sentences, **kwargs):
            if isinstance(sentences, str):
                return np.random.rand(384).astype(np.float32)
            else:
                return np.random.rand(len(sentences), 384).astype(np.float32)
                
        self.mock_model_instance = MagicMock()
        self.mock_model_instance.encode.side_effect = mock_encode
        self.mock_model_instance.max_seq_length = 256
        self.mock_encoder.return_value = self.mock_model_instance

    def tearDown(self):
        self.mock_encoder_patch.stop()
        self.temp_dir.cleanup()

    def test_semantic_search_init(self):
        ss = SemanticSearch()
        self.mock_encoder.assert_called_once_with("all-MiniLM-L6-v2")
        self.assertIsNone(ss.embeddings)
        self.assertIsNone(ss.documents)
        self.assertEqual(ss.document_map, {})

    def test_generate_embedding(self):
        ss = SemanticSearch()
        embedding = ss.generate_embedding("Hello world")
        self.assertEqual(embedding.shape, (384,))
        self.mock_model_instance.encode.assert_called_with("Hello world", convert_to_numpy=True)

    def test_build_and_load_embeddings(self):
        ss = SemanticSearch()
        documents = {
            "doc1": "First document description",
            "doc2": "Second document description",
            "doc3": "Third document description"
        }
        
        # Build embeddings and verify files are created
        embeddings = ss.build_embeddings(documents, self.temp_dir.name)
        self.assertEqual(embeddings.shape, (3, 384))
        self.assertEqual(ss.document_map, {0: "doc1", 1: "doc2", 2: "doc3"})
        self.assertEqual(ss.documents, documents)
        
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "embeddings.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "document_map.pkl")))
        
        # Create a new instance and load from cache
        ss2 = SemanticSearch()
        
        # Should have loaded from files (mock model encode should not be called this time)
        self.mock_model_instance.encode.reset_mock()
        ss2.load_or_create_embeddings(documents, self.temp_dir.name)
        self.mock_model_instance.encode.assert_not_called()
        
        self.assertEqual(ss2.embeddings.shape, (3, 384))
        self.assertEqual(ss2.document_map, {0: "doc1", 1: "doc2", 2: "doc3"})

    def test_search_correctness(self):
        ss = SemanticSearch()
        documents = {
            "doc1": "Bear",
            "doc2": "Space",
            "doc3": "Adventure"
        }
        ss.build_embeddings(documents, self.temp_dir.name)
        
        # Run search with normal limit
        results = ss.search("query", limit=2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(res in documents for res in results))

    def test_search_out_of_bounds_limit(self):
        ss = SemanticSearch()
        documents = {
            "doc1": "Bear",
            "doc2": "Space"
        }
        ss.build_embeddings(documents, self.temp_dir.name)
        
        # Run search with limit larger than document count
        # Top-k safety should cap it and prevent RuntimeError
        results = ss.search("query", limit=5)
        self.assertEqual(len(results), 2)

    def test_search_empty_embeddings(self):
        ss = SemanticSearch()
        results = ss.search("query")
        self.assertEqual(results, [])

    def test_cosine_similarity(self):
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        vec3 = np.array([0.0, 1.0, 0.0])
        
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 1.0)
        self.assertAlmostEqual(cosine_similarity(vec1, vec3), 0.0)
        
        # Zero norm fallback
        self.assertEqual(cosine_similarity(vec1, np.array([0.0, 0.0, 0.0])), 0.0)

    @patch('sys.stdout')
    def test_verify_model(self, mock_stdout):
        verify_model()
        self.mock_encoder.assert_called_with('all-MiniLM-L6-v2')

    @patch('sys.stdout')
    def test_embed_text(self, mock_stdout):
        embed_text("test text")
        self.mock_model_instance.encode.assert_called_with("test text", convert_to_numpy=True)

    @patch('sys.stdout')
    def test_embed_query(self, mock_stdout):
        embed_query("test query")
        self.mock_model_instance.encode.assert_called_with("test query", convert_to_numpy=True)
