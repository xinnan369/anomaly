"""All HTTP, route checks and signing operations use mocks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools import download_policy as policy


class TransferTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.destination = self.root / "example.bin"
        self.item = {
            "name": self.destination.name, "version": "revision-1", "size_bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(), "purpose": "offline test fixture",
            "mirror": "https://hf-mirror.com/fixture", "redirect_chain": [], "route_status": "verified",
            "destination": str(self.destination), "file_budget_bytes": 3, "direct_interface_index": 12,
        }
        self.plan = {"task_id": "TASK-004", "owner_approval_record": "mock-only",
                     "task_budget_bytes": 3, "files": [self.item]}
        self.addCleanup(patch.stopall)
        patch.object(policy, "TRUST_DIRECTORY", self.root).start()
        patch.object(policy, "CHUNK", 2).start()
        patch.object(policy.platform, "release", return_value="native").start()
        self.route = patch.object(policy, "verify_direct_route").start()
        self.session = patch.object(policy, "DirectSession").start()
        self.sign = patch.object(policy, "verify_signature").start()

    def raw(self):
        return json.dumps(self.plan).encode()

    def execute(self):
        policy.execute(self.raw(), self.root / "mock.sig")

    def assert_no_http(self):
        self.session.return_value.__enter__.return_value.get.assert_not_called()

    def test_missing_approval_refuses_before_any_request(self):
        with self.assertRaisesRegex(ValueError, "owner-signed"):
            policy.execute(self.raw(), None)
        self.session.assert_not_called()

    def test_invalid_owner_signature_refuses_before_any_request(self):
        self.sign.side_effect = ValueError("invalid owner signature")
        with self.assertRaisesRegex(ValueError, "invalid owner signature"):
            self.execute()
        self.session.assert_not_called()

    def test_unknown_size_generates_plan_and_blocks_execution(self):
        self.item["size_bytes"] = None
        self.assertTrue(policy.describe(self.plan)["owner_threshold_triggered"])
        with self.assertRaisesRegex(ValueError, "unknown size"):
            self.execute()
        self.session.assert_not_called()

    def test_prior_reservations_block_task_before_first_request(self):
        self.plan["task_budget_bytes"] = 5
        (self.root / "TASK-004.json").write_text('{"total":3,"files":{}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "task cumulative"):
            self.execute()
        self.session.assert_not_called()

    def test_prior_reservation_blocks_file_even_with_task_budget(self):
        self.plan["task_budget_bytes"] = 10
        ledger = policy.Ledger(self.root / "TASK-004.json", 10)
        ledger.charge(self.item["sha256"], 1, 3)
        with self.assertRaisesRegex(ValueError, "cumulative file"):
            self.execute()
        self.session.assert_not_called()

    def test_cumulative_threshold_across_individually_small_files(self):
        self.item["size_bytes"] = 600_000_000
        second = {**self.item, "destination": str(self.root / "second.bin"), "name": "second.bin", "sha256": "1" * 64}
        self.plan["files"].append(second)
        with patch.object(policy, "CHUNK", 64 * 1024 * 1024):
            report = policy.describe(self.plan)
        self.assertEqual(report["projected_task_bytes"], 1_200_000_000)
        self.assertTrue(report["owner_threshold_triggered"])
        self.session.assert_not_called()

    def test_foreign_hop_requires_distinct_owner_exception(self):
        self.item["redirect_chain"] = ["https://cas-bridge.xethub.hf.co/fixture"]
        with self.assertRaisesRegex(ValueError, "separate explicit owner"):
            self.execute()
        self.session.assert_not_called()

    def test_draft_route_blocks_before_http(self):
        self.item["route_status"] = "unverified"
        self.assertIsNone(policy.describe(self.plan)["files"][0]["final_source"])
        with self.assertRaisesRegex(ValueError, "redirect chain must be verified"):
            self.execute()
        self.session.assert_not_called()

    def test_foreign_exception_and_exact_hop_are_owner_bound(self):
        self.item["redirect_chain"] = ["https://cdn.invalid/signed-object"]
        self.item["owner_foreign_exception_record"] = "mock-explicit-separate-record"
        session = MagicMock()
        initial, final = MagicMock(), MagicMock()
        initial.__enter__.return_value = initial
        initial.is_redirect = True
        initial.headers = {"Location": "https://cdn.invalid/signed-object?session=one"}
        final.__enter__.return_value = final
        final.is_redirect = False
        session.get.side_effect = [initial, final]
        policy.approval(self.plan, policy.file_matrix(self.plan), self.raw(), self.root / "mock.sig")
        with policy.approved_request(session, self.item, {}, self.route) as result:
            self.assertIs(result, final)
        self.assertEqual([call.args[0] for call in self.route.call_args_list],
                         ["hf-mirror.com", "cdn.invalid"])

    def test_unexpected_redirect_is_not_followed(self):
        session = MagicMock()
        reply = session.get.return_value.__enter__.return_value
        reply.is_redirect = True
        reply.headers = {"Location": "https://unapproved.invalid/fixture"}
        self.item["redirect_chain"] = ["https://mirrors.ustc.edu.cn/fixture"]
        with self.assertRaisesRegex(ValueError, "unapproved redirect endpoint"):
            with policy.approved_request(session, self.item, {}, self.route):
                self.fail("unexpected endpoint accepted")
        session.get.assert_called_once()
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    def test_wrong_interface_blocks_before_http(self):
        self.route.side_effect = ValueError("route diverges")
        with self.assertRaisesRegex(ValueError, "route diverges"):
            self.execute()
        self.assert_no_http()

    def test_verified_local_cache_makes_no_network_requests(self):
        self.destination.write_bytes(b"abc")
        self.execute()
        self.assert_no_http()

    def test_resume_preserves_partial_and_reserves_only_missing_body(self):
        directory = policy.range_cache(self.destination, self.item["version"])
        directory.mkdir()
        (directory / "000000000000.download").write_bytes(b"a")
        (directory / "000000000002.part").write_bytes(b"c")
        self.assertEqual(policy.describe(self.plan)["projected_task_bytes"], 1)
        session = self.session.return_value.__enter__.return_value
        reply = session.get.return_value.__enter__.return_value
        reply.is_redirect = False
        reply.status_code = 206
        reply.headers = {"Content-Range": "bytes 1-1/3", "Content-Length": "1"}
        reply.iter_content.return_value = [b"b"]
        self.execute()
        self.assertEqual(self.destination.read_bytes(), b"abc")
        self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=1-1")
        state = policy.Ledger(self.root / "TASK-004.json", 3).state
        self.assertEqual(state["total"], 1)
        self.assertEqual([event["event"] for event in state["events"]],
                         ["range_reserved", "range_completed", "file_verified"])

    def test_failed_body_reservation_survives_retry_and_restart(self):
        session = self.session.return_value.__enter__.return_value
        reply = session.get.return_value.__enter__.return_value
        reply.is_redirect = False
        reply.status_code = 206
        reply.headers = {"Content-Range": "bytes 0-1/3", "Content-Length": "2"}
        reply.iter_content.return_value = [b"a"]
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.execute()
        with self.assertRaisesRegex(ValueError, "task cumulative"):
            self.execute()
        session.get.assert_called_once()

    def test_offline_assembly_keeps_ranges_and_rejects_missing_ones(self):
        directory = policy.range_cache(self.destination, self.item["version"])
        directory.mkdir()
        first = directory / "000000000000.part"
        first.write_bytes(b"ab")
        with self.assertRaisesRegex(ValueError, "complete local range"):
            policy.assemble_local(self.destination, self.item["version"], 3, self.item["sha256"])
        (directory / "000000000002.part").write_bytes(b"c")
        policy.assemble_local(self.destination, self.item["version"], 3, self.item["sha256"])
        self.assertEqual(self.destination.read_bytes(), b"abc")
        self.assertTrue(first.exists())
        self.session.assert_not_called()

    def test_direct_session_does_not_use_proxy_or_drain_redirects(self):
        patch.stopall()
        with policy.DirectSession() as direct:
            self.assertFalse(direct.trust_env)
            self.assertEqual(list(direct.resolve_redirects(None, None)), [])
