"""Dataset schema smoke tests (no live APIs)."""

from __future__ import annotations

import unittest
from pathlib import Path

from eval.runner import DEFAULT_DATASET, load_dataset


class TestEvalDataset(unittest.TestCase):
    def test_dataset_loads_twenty_cases(self) -> None:
        dataset = load_dataset(DEFAULT_DATASET)
        self.assertEqual(dataset.version, "1")
        self.assertEqual(len(dataset.cases), 20)
        ids = {c.id for c in dataset.cases}
        self.assertEqual(len(ids), 20)

    def test_every_case_has_query_and_expect(self) -> None:
        dataset = load_dataset(DEFAULT_DATASET)
        for case in dataset.cases:
            self.assertTrue(case.query.strip())
            self.assertTrue(case.description.strip())
            self.assertIn(case.mode, ("parse", "full"))


if __name__ == "__main__":
    unittest.main()
