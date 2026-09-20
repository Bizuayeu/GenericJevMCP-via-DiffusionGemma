import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import service

class ServiceTests(unittest.TestCase):
    def test_adapter_starts_without_private_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=root/'state'
            state.mkdir()
            snapshot=str(Path.home()/'.cache/huggingface/hub/model/snapshots/revision')
            (state/'download-status.json').write_text(json.dumps({'models':{'diffusion':{'snapshot':snapshot}}}))
            for name in ['server.py','decision.py','media.py','corpus.py']:
                (root/name).write_text('# fixture')
            with patch.object(service,'ROOT',root),patch.object(service,'STATE',state), \
                 patch('service.run',side_effect=['','container']) as run:
                service.start('adapter')
            args=run.call_args.args
            self.assertNotIn('--corpus',args)
            self.assertIn('NVIDIA_VISIBLE_DEVICES=void',args)
