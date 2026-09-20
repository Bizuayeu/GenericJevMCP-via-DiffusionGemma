import subprocess
import sys
import unittest
from pathlib import Path
from jev import client
from scripts import prepare_model, service

ROOT=Path(__file__).resolve().parents[1]

class EntrypointTests(unittest.TestCase):
    def test_package_and_operational_commands_start_from_checkout(self):
        for args in [('-m','jev.client','--help'),('-m','jev.calibration','--help'),
                     ('scripts/service.py','--help'),('scripts/prepare_model.py','--help')]:
            with self.subTest(args=args):
                result=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('usage:',result.stdout)
    def test_local_state_stays_at_checkout_root(self):
        self.assertEqual(client.ROOT,ROOT)
        self.assertEqual(service.ROOT,ROOT)
        self.assertEqual(prepare_model.ROOT,ROOT)
