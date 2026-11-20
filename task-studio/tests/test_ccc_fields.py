import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_app.prepare_ccc_fields import prepare
from task_app.assets import Assets
from task_app.export import generate


class CCCFieldsTests(unittest.TestCase):
    def test_padding_preserves_geometry_orientation_and_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            # A 6-high, 4-wide image: bottom brighter than top. No resizing needed.
            values = np.repeat(np.arange(6, dtype=np.uint8)[:, None] * 30, 4, axis=1)
            path = folder/'source.png'
            Image.fromarray(values).save(path)
            spec = dict(id='example', name='Intensity example', file=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), original_dataset='Test fixture')
            prepared = prepare(spec, folder, folder/'prepared')
            self.assertEqual(prepared.shape, (6,6))
            np.testing.assert_array_equal(prepared[:,1:5], values[::-1])
            self.assertTrue(np.all(prepared[:,[0,5]] == 0))
            self.assertEqual(prepared.mean(), 50)
            assets = Assets(folder/'prepared')
            self.assertFalse(assets.dataset_warnings)
            result = generate(dict(datasets=['example'], colormaps=['gray'], count=1, seed=42,
                                   operation=['identity'], selection=['field'], aggregation=['mean'], action=['retrieve']),
                              assets, folder/'batch')
            self.assertEqual(result['records'][0]['ground_truth'], 50)
            meta = json.loads((folder/'prepared/example.json').read_text())
            self.assertEqual(meta['provenance']['padding_top_bottom_left_right'], [0,0,1,1])
            self.assertIn('not original physical values', meta['provenance']['scalar_units'])

    def test_rejects_changed_source_and_non_gray_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder/'color.png'
            Image.new('RGB', (4,4), 'red').save(path)
            spec = dict(id='example', name='Example', file=path.name, sha256='wrong')
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare(spec, folder, folder/'prepared')
            spec['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'near-gray'):
                prepare(spec, folder, folder/'prepared')


if __name__ == '__main__': unittest.main()
