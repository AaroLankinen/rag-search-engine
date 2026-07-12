"""Unit tests for the rerank methods in hybrid_search_cli.py.

Tests cover JSON parsing logic, bracket extraction, result ordering,
cross-encoder score sorting, and fallback behavior. Integration tests
that require heavy imports (sentence_transformers, OpenAI) are
excluded—those are validated via the bootdev CLI tests.
"""

import json
import unittest


class TestBatchJsonParsing(unittest.TestCase):
    """Tests for JSON parsing logic used in batch reranking.

    The batch rerank method extracts a JSON array from an LLM response
    by locating the first '[' and last ']' then parsing the substring.
    """

    def _extract_ids(self, raw_text):
        """Replicate the bracket-extraction logic from the CLI.

        Args:
            raw_text: The raw LLM response string.

        Returns:
            A list of ints if successful, or None if extraction fails.
        """
        bracket_start = raw_text.find("[")
        bracket_end = raw_text.rfind("]")
        if bracket_start == -1 or bracket_end == -1:
            return None
        json_str = raw_text[bracket_start : bracket_end + 1]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return parsed
        return None

    def test_parse_clean_json_array(self):
        """A clean JSON array should be parsed correctly."""
        self.assertEqual(self._extract_ids("[1, 2, 3]"), [1, 2, 3])

    def test_parse_json_with_surrounding_text(self):
        """JSON extraction should work even with surrounding text."""
        raw = "Here is the ranking:\n[42, 7, 13]\nThat's my answer."
        self.assertEqual(self._extract_ids(raw), [42, 7, 13])

    def test_parse_no_brackets_returns_none(self):
        """When no brackets are found, parsing should fail gracefully."""
        self.assertIsNone(self._extract_ids("I cannot rank these movies."))

    def test_parse_invalid_json_inside_brackets(self):
        """Malformed JSON inside brackets should return None."""
        self.assertIsNone(self._extract_ids("[not valid json]"))

    def test_parse_non_integer_array(self):
        """An array of non-integers should return None (ints only)."""
        self.assertIsNone(self._extract_ids('["a", "b", "c"]'))

    def test_parse_empty_array(self):
        """An empty JSON array should be parsed as an empty list."""
        self.assertEqual(self._extract_ids("[]"), [])

    def test_parse_single_element(self):
        """A single-element array should be parsed correctly."""
        self.assertEqual(self._extract_ids("[42]"), [42])

    def test_parse_markdown_wrapped_json(self):
        """JSON wrapped in markdown code fences should still be extracted."""
        raw = "```json\n[10, 20, 30]\n```"
        self.assertEqual(self._extract_ids(raw), [10, 20, 30])

    def test_parse_mixed_types_returns_none(self):
        """An array with mixed int and string types should return None."""
        self.assertIsNone(self._extract_ids('[1, "two", 3]'))

    def test_parse_floats_returns_none(self):
        """An array of floats should return None (only ints are valid)."""
        self.assertIsNone(self._extract_ids("[1.5, 2.5, 3.5]"))


class TestBatchResultOrdering(unittest.TestCase):
    """Tests the result ordering logic used after LLM ranking.

    The batch method builds an ordered list by:
    1. Following the LLM's ranked ID order (filtering to valid IDs)
    2. Appending any remaining results not mentioned by the LLM
    3. Truncating to the requested limit
    """

    def _order_results(self, ranked_ids, results, limit):
        """Replicate the ordering logic from the CLI.

        Args:
            ranked_ids: List of movie IDs in LLM-ranked order.
            results: List of RRF result dicts.
            limit: Maximum number of results to return.

        Returns:
            List of result dicts in final display order.
        """
        result_by_id = {int(r["document"]["id"]): r for r in results}

        seen = set()
        ordered = []
        for doc_id in ranked_ids:
            if doc_id in result_by_id and doc_id not in seen:
                seen.add(doc_id)
                ordered.append(result_by_id[doc_id])

        for res in results:
            doc_id = int(res["document"]["id"])
            if doc_id not in seen:
                seen.add(doc_id)
                ordered.append(res)

        return ordered[:limit]

    def _make_result(self, doc_id):
        """Create a minimal mock result dict.

        Args:
            doc_id: The movie document ID.

        Returns:
            A dict matching the rrf_search result format.
        """
        return {
            "document": {"id": doc_id, "title": f"Movie {doc_id}", "description": f"Desc {doc_id}"},
            "rrf_score": 0.03,
            "bm25_rank": 1,
            "semantic_rank": 1,
        }

    def test_llm_ordering_respected(self):
        """Results should follow the LLM's ranking order."""
        results = [self._make_result(i) for i in [1, 2, 3]]
        ordered = self._order_results([3, 1, 2], results, 3)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids, [3, 1, 2])

    def test_missing_ids_appended(self):
        """IDs not in the LLM response should appear after ranked ones."""
        results = [self._make_result(i) for i in [1, 2, 3, 4]]
        ordered = self._order_results([2, 4], results, 4)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids[:2], [2, 4])
        self.assertIn(1, ids[2:])
        self.assertIn(3, ids[2:])

    def test_limit_truncation(self):
        """Output should be truncated to the requested limit."""
        results = [self._make_result(i) for i in [1, 2, 3, 4, 5]]
        ordered = self._order_results([5, 4, 3, 2, 1], results, 3)
        self.assertEqual(len(ordered), 3)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids, [5, 4, 3])

    def test_unknown_ids_ignored(self):
        """IDs from the LLM that don't match any result should be ignored."""
        results = [self._make_result(i) for i in [1, 2]]
        ordered = self._order_results([99, 1, 2], results, 2)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids, [1, 2])

    def test_duplicate_ids_handled(self):
        """Duplicate IDs from the LLM should only appear once."""
        results = [self._make_result(i) for i in [1, 2, 3]]
        ordered = self._order_results([2, 2, 1, 1, 3], results, 3)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids, [2, 1, 3])

    def test_empty_llm_response_uses_original_order(self):
        """An empty LLM response should fall back to original order."""
        results = [self._make_result(i) for i in [1, 2, 3]]
        ordered = self._order_results([], results, 3)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids, [1, 2, 3])

    def test_fallback_on_none_uses_original(self):
        """When ranked_ids is the fallback (all original IDs), order is preserved."""
        results = [self._make_result(i) for i in [10, 20, 30]]
        # Simulate fallback: ranked_ids = original order
        fallback_ids = [int(r["document"]["id"]) for r in results]
        ordered = self._order_results(fallback_ids, results, 3)
        ids = [int(r["document"]["id"]) for r in ordered]
        self.assertEqual(ids, [10, 20, 30])


class TestCrossEncoderResultOrdering(unittest.TestCase):
    """Tests the cross-encoder score-based sorting logic.

    The cross_encoder method attaches a cross_encoder_score to each result
    and sorts by (cross_encoder_score, rrf_score, -doc_id) descending.
    """

    def _make_result(self, doc_id, cross_score, rrf_score=0.03):
        """Create a mock result dict with a cross-encoder score.

        Args:
            doc_id: The movie document ID.
            cross_score: The cross-encoder relevance score.
            rrf_score: The RRF score (default 0.03).

        Returns:
            A dict matching the cross_encoder rerank result format.
        """
        return {
            "document": {"id": doc_id, "title": f"Movie {doc_id}", "description": f"Desc {doc_id}"},
            "rrf_score": rrf_score,
            "bm25_rank": 1,
            "semantic_rank": 1,
            "cross_encoder_score": cross_score,
        }

    def _sort_results(self, results):
        """Replicate the cross-encoder sorting logic from the CLI.

        Args:
            results: List of result dicts with cross_encoder_score.

        Returns:
            Sorted list of result dicts.
        """
        return sorted(
            results,
            key=lambda x: (x["cross_encoder_score"], x["rrf_score"], -int(x["document"]["id"])),
            reverse=True,
        )

    def test_sorts_by_cross_encoder_score_descending(self):
        """Results should be sorted by cross_encoder_score, highest first."""
        results = [
            self._make_result(1, 1.5),
            self._make_result(2, 3.7),
            self._make_result(3, -0.5),
        ]
        sorted_res = self._sort_results(results)
        scores = [r["cross_encoder_score"] for r in sorted_res]
        self.assertEqual(scores, [3.7, 1.5, -0.5])

    def test_tiebreak_by_rrf_score(self):
        """Equal cross-encoder scores should be broken by rrf_score."""
        results = [
            self._make_result(1, 2.0, rrf_score=0.01),
            self._make_result(2, 2.0, rrf_score=0.05),
        ]
        sorted_res = self._sort_results(results)
        ids = [int(r["document"]["id"]) for r in sorted_res]
        self.assertEqual(ids, [2, 1])

    def test_tiebreak_by_doc_id_ascending(self):
        """Equal scores should further be broken by doc ID ascending."""
        results = [
            self._make_result(10, 2.0, rrf_score=0.03),
            self._make_result(5, 2.0, rrf_score=0.03),
        ]
        sorted_res = self._sort_results(results)
        ids = [int(r["document"]["id"]) for r in sorted_res]
        # Lower doc ID first (ascending) when scores are equal
        self.assertEqual(ids, [5, 10])

    def test_negative_scores_handled(self):
        """Cross-encoder scores can be negative; sorting should still work."""
        results = [
            self._make_result(1, -3.0),
            self._make_result(2, -1.0),
            self._make_result(3, -5.0),
        ]
        sorted_res = self._sort_results(results)
        scores = [r["cross_encoder_score"] for r in sorted_res]
        self.assertEqual(scores, [-1.0, -3.0, -5.0])

    def test_limit_applied_after_sorting(self):
        """Only the top N results should be kept after sorting."""
        results = [self._make_result(i, float(i)) for i in range(1, 6)]
        sorted_res = self._sort_results(results)[:3]
        self.assertEqual(len(sorted_res), 3)
        ids = [int(r["document"]["id"]) for r in sorted_res]
        self.assertEqual(ids, [5, 4, 3])

    def test_single_result(self):
        """A single result should be returned unchanged."""
        results = [self._make_result(42, 7.5)]
        sorted_res = self._sort_results(results)
        self.assertEqual(len(sorted_res), 1)
        self.assertEqual(int(sorted_res[0]["document"]["id"]), 42)

    def test_empty_results(self):
        """An empty result list should return an empty list."""
        sorted_res = self._sort_results([])
        self.assertEqual(sorted_res, [])


if __name__ == "__main__":
    unittest.main()
