import tempfile
import unittest
from pathlib import Path

from morango_lmfcn.manifest import prepare, read_manifest, validate_splits


class ManifestTest(unittest.TestCase):
    def test_split_is_stratified_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "Dataset" / "Nusrat Sultana" / "Original"
            for directory in ("FreshStrawberry", "RottenStrawberry"):
                folder = source / directory
                folder.mkdir(parents=True)
                for index in range(20):
                    (folder / f"image_{index}.jpg").touch()
            output = root / "output"
            counts = prepare(root / "Dataset", output, seed=42)
            self.assertEqual(counts["train"][0], 14)
            self.assertEqual(counts["val"][1], 3)
            self.assertEqual(counts["test"][0], 3)
            splits = [read_manifest(output / f"{name}.txt") for name in ("train", "val", "test")]
            validate_splits(*splits)
            with self.assertRaises(ValueError):
                validate_splits(splits[0], splits[0], splits[2])


if __name__ == "__main__":
    unittest.main()
