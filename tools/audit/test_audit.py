"""Offline regression checks for manifest validation and HTTP download bounds."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from manifest import CONFIGURATIONS, check_manifest
from sources import ARCHIVES, MANIFEST, read_https, verified_resource


def rows(object_id):
    return [
        {
            "img_path": f"Bird/Good/{object_id}/A{view}_I{light}.png",
            "view": view, "illumination": light, "cls_name": "Bird",
            "object_name": object_id, "object_anomaly": 0, "image_anomaly": 0,
        }
        for view, light in sorted(CONFIGURATIONS)
    ]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.data = {split: {cls: [] for cls in ARCHIVES} for split in ("train", "test")}
        self.data["train"]["Bird"] = rows("001")
        self.data["test"]["Bird"] = rows("000")

    def test_counts_complete_objects(self):
        result = check_manifest(self.data)
        self.assertEqual(result["total"], {"objects": 2, "images": 240})
        self.assertEqual(result["classes"]["Bird"]["train_good"], {"objects": 1, "images": 120})
        self.assertEqual(result["overlaps"]["full_keys"], 0)

    def test_rejects_incomplete_object(self):
        self.data["test"]["Bird"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete object"):
            check_manifest(self.data)

    def test_rejects_cross_split_instance(self):
        self.data["test"]["Bird"] = rows("001")
        with self.assertRaisesRegex(ValueError, "cross-split overlap"):
            check_manifest(self.data)

    def test_rejects_duplicate_configuration(self):
        self.data["train"]["Bird"].append(self.data["train"]["Bird"][0])
        with self.assertRaisesRegex(ValueError, "duplicate path"):
            check_manifest(self.data)

    def test_rejects_swapped_official_split(self):
        self.data["train"]["Bird"], self.data["test"]["Bird"] = (
            self.data["test"]["Bird"], self.data["train"]["Bird"],
        )
        with self.assertRaisesRegex(ValueError, "published split differs from generator"):
            check_manifest(self.data)


class ResourceTests(unittest.TestCase):
    def test_rejects_corrupt_cached_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / MANIFEST.name).write_bytes(b"wrong release")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verified_resource(cache, MANIFEST)

    def test_rejects_non_range_response_before_reading(self):
        with patch("sources.requests.get") as get:
            response = get.return_value.__enter__.return_value
            response.status_code = 200
            with self.assertRaisesRegex(ValueError, "ignored the byte range"):
                read_https("https://example.invalid/test.zip", 10, (0, 9, 100))
            response.iter_content.assert_not_called()

    def test_rejects_unexpected_content_range_before_reading(self):
        with patch("sources.requests.get") as get:
            response = get.return_value.__enter__.return_value
            response.status_code = 206
            response.headers = {"Content-Range": "bytes 1-10/100"}
            with self.assertRaisesRegex(ValueError, "unexpected Content-Range"):
                read_https("https://example.invalid/test.zip", 10, (0, 9, 100))
            response.iter_content.assert_not_called()

    def test_rejects_oversize_chunk_without_content_length(self):
        with patch("sources.requests.get") as get:
            response = get.return_value.__enter__.return_value
            response.headers = {}
            response.iter_content = Mock(return_value=iter([b"elevenbytes"]))
            with self.assertRaisesRegex(ValueError, "exceeds byte limit"):
                read_https("https://example.invalid/test.zip", 10)


if __name__ == "__main__":
    unittest.main()
