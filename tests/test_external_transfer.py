"""Regression for short HTTP reads and publisher Git/LFS identity verification."""

import hashlib
import io
import unittest
from unittest.mock import patch

from scripts.review_external_shoulder_sources import get_bytes
from scripts.verify_external_source_integrity import matches


class ChunkedResponse(io.BytesIO):
  def __init__(self, payload, expected=None):
    super().__init__(payload)
    self.headers = {} if expected is None else {"Content-Length": str(expected)}

  def read(self, size=-1):
    return super().read(min(size, 2))


class TransferTests(unittest.TestCase):
  def test_short_reads_are_joined_to_full_payload(self):
    with patch("urllib.request.urlopen", return_value=ChunkedResponse(b"1234567", 7)):
      self.assertEqual(get_bytes("https://example.test/file", 20), b"1234567")

  def test_truncated_http_body_rejected(self):
    with patch("urllib.request.urlopen", return_value=ChunkedResponse(b"123", 7)):
      with self.assertRaisesRegex(ValueError, "truncated"):
        get_bytes("https://example.test/file", 20)

  def test_known_and_unknown_size_limits_enforced(self):
    for expected in (10, None):
      with patch(
        "urllib.request.urlopen", return_value=ChunkedResponse(b"1234567890", expected)
      ):
        with self.assertRaisesRegex(ValueError, "per-file"):
          get_bytes("https://example.test/file", 5)

  def test_publisher_lfs_hash_and_length_required(self):
    payload = b"motion"
    metadata = {
      "size": len(payload),
      "lfs": {"sha256": hashlib.sha256(payload).hexdigest()},
    }
    self.assertTrue(matches(payload, metadata))
    self.assertFalse(matches(b"motio", metadata))
    self.assertFalse(matches(b"motiox", metadata))

  def test_publisher_git_blob_hash_has_git_header(self):
    payload = b"documentation"
    metadata = {
      "size": len(payload),
      "blobId": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
    }
    self.assertTrue(matches(payload, metadata))
    self.assertFalse(matches(b"DocumentatioN", metadata))


if __name__ == "__main__":
  unittest.main()
