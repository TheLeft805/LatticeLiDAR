import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import model_config
from trusted_assets import verify_release_assets


class TrustedAssets(unittest.TestCase):
    def test_copies_allowed_and_same_length_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, heads, manifest = root / 'copy.pt', root / 'copy.npz', root / 'ASSETS.json'
            checkpoint.write_bytes(b'weights')
            heads.write_bytes(b'heads')
            manifest.write_text(json.dumps({'files': {
                'weights/backbone.pt': hashlib.sha256(b'weights').hexdigest(),
                'weights/reconstruction.npz': hashlib.sha256(b'heads').hexdigest(),
            }}))
            verify_release_assets(checkpoint, heads, manifest_path=manifest)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Untrusted'):
                verify_release_assets(checkpoint, heads, manifest_path=manifest)

    def test_missing_allowlist_digest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'ASSETS.json'
            manifest.write_text('{"files": {}}')
            with self.assertRaisesRegex(ValueError, 'Missing trusted'):
                verify_release_assets('unused.pt', 'unused.npz', manifest_path=manifest)

    def test_model_is_not_imported_when_asset_verification_fails(self):
        args = SimpleNamespace(checkpoint='untrusted.pt', heads='heads.npz', quantization_mm=16)
        original_import = __import__
        imported = []
        def track(name, *args, **kwargs):
            imported.append(name)
            return original_import(name, *args, **kwargs)
        with patch('trusted_assets.verify_release_assets', side_effect=ValueError('untrusted')):
            with patch('builtins.__import__', side_effect=track):
                with self.assertRaisesRegex(ValueError, 'untrusted'):
                    model_config.create_codec(args)
        self.assertNotIn('reconstruction_codec', imported)


if __name__ == '__main__':
    unittest.main()
