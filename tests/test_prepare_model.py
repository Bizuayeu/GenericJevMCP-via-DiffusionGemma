import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.prepare_model import verify_snapshot

class SnapshotTests(unittest.TestCase):
    def test_snapshot_requires_revision_cache_and_exact_file_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp)
            path=home/'.cache/huggingface/hub/model/snapshots/revision'
            path.mkdir(parents=True)
            (path/'weights').write_bytes(b'abc')
            lock={'revision':'revision','files':[{'path':'weights','bytes':3}]}
            with patch('scripts.prepare_model.Path.home',return_value=home):
                self.assertEqual(verify_snapshot(path,lock)['status'],'complete')
                (path/'weights').write_bytes(b'ab')
                with self.assertRaises(ValueError): verify_snapshot(path,lock)
                with self.assertRaises(ValueError): verify_snapshot(home,lock)
