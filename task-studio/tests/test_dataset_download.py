import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_app.assets import Assets
from task_app.dataset_download import (Budget, DownloadLimitError, HTTPRangeFile,
    SOURCES, bounded_get, contact_sheet, main, scalar_snapshot, write_prepared)
from task_app.resampling import downsample
from task_app.export import generate


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class Array:
    def __init__(self, values, **attrs):
        self.values = np.asarray(values)
        self.attrs = attrs
        self.shape, self.ndim = self.values.shape, self.values.ndim
    def __getitem__(self, key): return self.values[key]


class Group(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}


class DatasetDownloadTests(unittest.TestCase):
    def test_fractional_averaging_preserves_mean_and_bounds(self):
        a = np.arange(101 * 101, dtype=float).reshape(101, 101)
        result = downsample(a)
        self.assertEqual(result.shape, (100, 100))
        self.assertAlmostEqual(result.mean(), a.mean())
        self.assertGreaterEqual(result.min(), a.min())
        self.assertLessEqual(result.max(), a.max())
        np.testing.assert_allclose(downsample(np.full((201, 201), 7)), 7)
        checker = np.indices((200, 200)).sum(axis=0) % 2
        np.testing.assert_allclose(downsample(checker), .5)  # Decimation would alias to a constant 0 or 1.
        small = np.arange(20 * 20).reshape(20, 20)
        np.testing.assert_array_equal(downsample(small), small)
        self.assertEqual(downsample(np.ones((200, 400))).shape, (50, 100))
        with self.assertRaises(ValueError): downsample(np.full((3, 3), np.nan))

    def test_range_file_seeks_caches_and_enforces_limits(self):
        payload = bytes(range(256)) * 1000
        requests = []
        def opener(request, timeout):
            first, last = map(int, request.headers['Range'][6:].split('-'))
            requests.append((first, last))
            return Response(payload[first:last+1], 206,
                            {'Content-Range': f'bytes {first}-{last}/{len(payload)}'})
        budget = Budget(200000)
        with HTTPRangeFile('https://example.test/file', len(payload), budget, limit=131072, opener=opener) as file:
            file.seek(65530)
            self.assertEqual(file.read(20), payload[65530:65550])
            file.seek(100)
            self.assertEqual(file.read(30), payload[100:130])
            self.assertEqual(len(requests), 2)
            self.assertEqual(budget.used, 131072)
            file.seek(-1, 2)
            with self.assertRaises(DownloadLimitError): file.read(1)
        self.assertEqual(len(requests), 2)

    def test_refuses_full_download_and_misleading_ranges(self):
        for status, headers in [(200, {}), (206, {'Content-Range': 'bytes 1-9/1000000'})]:
            response = Response(b'x' * 100, status, headers)
            with HTTPRangeFile('https://example.test/file', 1000000, Budget(200000),
                               opener=lambda *a, **k: response) as file:
                with self.assertRaises(ValueError): file.read(10)
        with HTTPRangeFile('https://example.test/file', 1000000, Budget(100000),
                           opener=lambda *a, **k: self.fail('Must not request')) as file:
            with self.assertRaises(DownloadLimitError): file.read(20000000)

    def test_bounded_get_never_reads_above_budget(self):
        budget = Budget(20)
        response = Response(b'x' * 100, headers={'Content-Length': '100'})
        with patch('task_app.dataset_download.urlopen', return_value=response):
            with self.assertRaises(DownloadLimitError): bounded_get('https://example.test/a', 200, budget)
        self.assertEqual(budget.used, 20)

    def test_snapshot_selects_one_time_and_correct_orientation(self):
        values = np.arange(2 * 5 * 4 * 4).reshape(2, 5, 4, 4)
        file = Group(attrs={'n_spatial_dims': 2})
        file['t0_fields'] = Group(pressure=Array(values, sample_varying=True,
                                               time_varying=True, dim_varying=[True, True]))
        file['dimensions'] = Group(x=Array([0, 1, 2, 3]), y=Array([3, 2, 1, 0]),
                                   attrs={'spatial_dims': [b'x', b'y']})
        source = dict(field_candidates=['pressure'], expected_shape=[4, 4])
        result, metadata = scalar_snapshot(file, source)
        np.testing.assert_array_equal(result, values[0, 2].T[::-1])
        self.assertEqual(metadata['time_index'], 2)
        file.attrs['n_spatial_dims'] = 3
        with self.assertRaisesRegex(ValueError, 'two-spatial'): scalar_snapshot(file, source)

    def test_prepared_data_load_and_round_trip_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'datasets'
            source = dict(SOURCES[0], id='example', name='Example scalar')
            values = np.arange(144, dtype=float).reshape(12, 12)
            write_prepared(source, values, {'field': 'intensity'}, folder, 6)
            assets = Assets(folder)
            self.assertFalse(assets.dataset_warnings)
            self.assertEqual(assets.datasets['example']['values'].shape, (6, 6))
            self.assertEqual(assets.datasets['example']['source']['source_shape'], [12, 12])
            result = generate(dict(datasets=['example'], colormaps=['gray'], count=1, seed=42,
                                   operation=['identity'], selection=['field'], aggregation=['mean'], action=['retrieve']),
                              assets, Path(tmp)/'batch')
            self.assertAlmostEqual(result['records'][0]['ground_truth'], values.mean())
            self.assertEqual(result['records'][0]['provenance']['dataset_source']['citation'], source['citation'])
            (folder/'example.npy').write_bytes(b'corrupted')
            assets = Assets(folder)
            self.assertNotIn('example', assets.datasets)
            self.assertIn('checksum', assets.dataset_warnings[0])

    def test_main_writes_report_preview_and_reuses_verified_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = SOURCES[0]
            values = np.arange(512 * 512, dtype=float).reshape(512, 512)
            with patch('task_app.dataset_download.get_image', return_value=(values, {'fixture': True})) as fetch:
                self.assertEqual(main(['--only', source['id'], '--output', tmp]), 0)
                self.assertEqual(main(['--only', source['id'], '--output', tmp]), 0)
            self.assertEqual(fetch.call_count, 1)
            self.assertTrue((Path(tmp)/'preview.png').exists())
            report = json.loads((Path(tmp)/'download-report.json').read_text())
            self.assertEqual(report['prepared'], [source['id']])
            self.assertEqual(report['failures'], [])


if __name__ == '__main__':
    unittest.main()
