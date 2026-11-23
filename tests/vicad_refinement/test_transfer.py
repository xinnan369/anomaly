import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.vicad_syn.stage_refinement import get_part


MODULE = "tools.vicad_syn.stage_refinement"


class TransferTests(unittest.TestCase):
    def test_resume_verified_http_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp)
            partial = location / "000000000002.download"
            partial.write_bytes(b"ab")
            reply = MagicMock()
            reply.status_code = 206
            reply.headers = {"Content-Range": "bytes 4-6/9"}
            reply.iter_content.return_value = [b"cd", b"e"]
            with patch(MODULE + ".requests.get") as get:
                get.return_value.__enter__.return_value = reply
                output = get_part("mock-archive", 9, (2, 7), location)
            self.assertEqual(output.read_bytes(), b"abcde")
            self.assertFalse(partial.exists())
            self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=4-6")

    def test_does_not_accept_mismatched_content_range(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp)
            reply = MagicMock()
            reply.status_code = 206
            reply.headers = {"Content-Range": "bytes 1-5/9"}
            with (
                patch(MODULE + ".requests.get") as get,
                patch(MODULE + ".time.sleep"),
                patch(MODULE + ".print", create=True),
            ):
                get.return_value.__enter__.return_value = reply
                with self.assertRaisesRegex(RuntimeError, "exhausted retries"):
                    get_part("mock-archive", 9, (2, 7), location)
            self.assertEqual(list(location.iterdir()), [])
            reply.iter_content.assert_not_called()

    def test_preserves_completed_segment(self):
        with tempfile.TemporaryDirectory() as temp:
            location = Path(temp)
            final = location / "000000000002.part"
            final.write_bytes(b"abcde")
            with patch(MODULE + ".requests.get") as get:
                self.assertEqual(get_part("mock-archive", 9, (2, 7), location), final)
            get.assert_not_called()
