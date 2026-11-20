"""Public deployment boundaries exercised with independent browser clients."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_app.server import create_app
from task_app.assets import load_grid
from task_app.export import generate


def configuration(**overrides):
    return dict(datasets=['terrain_100'], colormaps=['gray'], operation=['identity'],
                selection=['field'], aggregation=['mean'], action=['retrieve'],
                count=1, seed=42) | overrides


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.app = create_app(dict(TESTING=True, SECRET_KEY='test-only-secret', OUTPUT_DIR=self.folder.name))
        self.state = self.app.extensions['task_runtime']
        self.client, self.other = self.app.test_client(), self.app.test_client()
        self.catalog = self.client.get('/api/catalog').get_json()
        self.other_catalog = self.other.get('/api/catalog').get_json()

    def tearDown(self):
        self.state.close()
        self.folder.cleanup()

    def post(self, path, data, client=None, **headers):
        client = client or self.client
        csrf = self.other_catalog['csrf_token'] if client is self.other else self.catalog['csrf_token']
        return client.post(path, json=data, headers={'X-CSRF-Token': csrf} | headers)

    def wait_for(self, ident, client=None):
        client = client or self.client
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = client.get('/api/jobs/' + ident)
            self.assertEqual(response.status_code, 200, response.data)
            result = response.get_json()
            if result['status'] not in ('queued', 'running'):
                return result
            time.sleep(.02)
        self.fail('Generation did not finish.')

    def test_full_upload_generation_download_and_isolation(self):
        upload = self.post('/api/upload', dict(kind='dataset', filename='own.csv',
                           content=base64.b64encode(b'1,2,3\n4,5,6\n7,8,9').decode(), extent=[0, 3]))
        self.assertEqual(upload.status_code, 200, upload.data)
        ident = upload.get_json()['id']
        self.assertNotIn(ident, [d['id'] for d in self.other.get('/api/catalog').get_json()['datasets']])
        cmap = self.post('/api/upload', dict(kind='colormap', filename='own.json',
                         content=base64.b64encode(b'["black","white"]').decode())).get_json()['id']
        self.assertNotIn(cmap, [c['id'] for c in self.other.get('/api/catalog').get_json()['colormaps']])
        cfg = configuration(datasets=[ident], colormaps=[cmap], aggregation=['mean', 'maximum'],
                            excluded_combinations=['identity|field|maximum|retrieve'])
        compatible = self.post('/api/compatibility', cfg).get_json()
        self.assertEqual((compatible['count'], compatible['compatible_count']), (1, 2))
        self.assertEqual(self.post('/api/generate', cfg, client=self.other).status_code, 400)
        response = self.post('/api/generate', cfg)
        self.assertEqual(response.status_code, 202, response.data)
        job = response.get_json()['id']
        result = self.wait_for(job)
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['records'][0]['ground_truth'], 5)
        self.assertEqual(result['records'][0]['aggregation'], 'mean')
        self.assertEqual(self.other.get('/api/jobs/' + job).status_code, 404)
        for filename in ['tasks.csv', 'tasks.md', 'tasks.json', 'manifest.json', 'images.zip', 'everything.zip', 'images/task_0001.png']:
            url = f'/downloads/{job}/{filename}'
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, filename)
            self.assertTrue(response.data)
            response.close()
            self.assertEqual(self.other.get(url).status_code, 404)
        response = self.client.get(f'/downloads/{job}/everything.zip')
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(json.loads(archive.read('tasks.json'))[0]['ground_truth'], 5)
            self.assertIn(f'inputs/{ident}.npy', archive.namelist())
        response.close()
        self.assertEqual(self.client.get(f'/downloads/{job}/../../pyproject.toml').status_code, 404)

    def test_csrf_origin_host_headers_and_upload_validation(self):
        cfg = configuration()
        self.assertEqual(self.client.post('/api/generate', json=cfg).status_code, 403)
        self.assertEqual(self.post('/api/generate', cfg, **{'X-CSRF-Token': self.other_catalog['csrf_token']}).status_code, 403)
        self.assertEqual(self.post('/api/generate', cfg, Origin='https://evil.example').status_code, 403)
        self.assertEqual(self.post('/api/generate', cfg, **{'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
        self.assertEqual(self.client.get('/api/catalog', headers={'Host': 'evil.example'}).status_code, 400)
        self.assertEqual(self.client.post('/api/upload', data='{}', headers={'X-CSRF-Token': self.catalog['csrf_token']}).status_code, 415)
        self.assertEqual(self.post('/api/upload', dict(kind='dataset', filename='own.csv', content='bad')).status_code, 400)
        self.assertEqual(self.post('/api/upload', {}).status_code, 400)
        self.assertEqual(self.client.post('/api/upload', data=' '*4_000_001, content_type='application/json',
                                         headers={'X-CSRF-Token': self.catalog['csrf_token']}).status_code, 413)
        response = self.client.get('/')
        self.assertIn(b'Scalar Task Studio', response.data)
        self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        response.close()
        for url in ['/app.js', '/style.css', '/combinations-vis.js']:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            response.close()
        self.assertEqual(self.app.test_client().get('/api/jobs/unknown').status_code, 401)

    def test_public_https_proxy_and_health(self):
        self.app.config.update(PUBLIC_BASE_URL='https://studio.example', TRUSTED_HOSTS=['studio.example'],
                               SESSION_COOKIE_SECURE=True)
        client = self.app.test_client()
        # Render terminates TLS: requests arrive as HTTP, with the public Host.
        response = client.get('/api/catalog', base_url='http://studio.example')
        self.assertEqual(response.status_code, 200)
        cookie = response.headers['Set-Cookie']
        for value in ['Secure', 'HttpOnly', 'SameSite=Lax']:
            self.assertIn(value, cookie)
        csrf = response.get_json()['csrf_token']
        result = client.post('/api/compatibility', json=configuration(), base_url='https://studio.example',
                             headers={'Origin': 'https://studio.example', 'X-CSRF-Token': csrf})
        self.assertEqual(result.status_code, 200, result.data)
        health = self.app.test_client().get('/healthz', base_url='http://studio.example')
        self.assertEqual(health.get_json(), {'status': 'ok'})
        self.assertNotIn('Set-Cookie', health.headers)

    def test_queue_single_generation_and_fairness(self):
        started, release = threading.Event(), threading.Event()
        seen = []
        def blocked(config, assets, output, progress, **kwargs):
            seen.append(config['seed'])
            started.set()
            release.wait(5)
            return generate(config, assets, output, progress, **kwargs)
        with patch('task_app.runtime.generate', blocked):
            try:
                first = self.post('/api/generate', configuration(seed=1)).get_json()['id']
                self.assertTrue(started.wait(2))
                second = self.post('/api/generate', configuration(seed=2), client=self.other).get_json()['id']
                self.assertEqual(self.other.get('/api/jobs/'+second).get_json()['status'], 'queued')
                self.assertEqual(seen, [1])
                self.assertEqual(self.post('/api/generate', configuration()).status_code, 429)
                release.set()
                self.assertEqual(self.wait_for(first)['status'], 'complete')
                self.assertEqual(self.wait_for(second, self.other)['status'], 'complete')
                self.assertEqual(seen, [1, 2])
            finally:
                release.set()

    def test_expiration_removes_files_and_session(self):
        ident = self.post('/api/generate', configuration()).get_json()['id']
        self.assertEqual(self.wait_for(ident)['status'], 'complete')
        path = Path(self.folder.name)/ident
        self.assertTrue(path.is_dir())
        with self.state.lock:
            self.state.jobs[ident]['finished'] -= self.app.config['JOB_TTL']+1
        self.state.cleanup()
        self.assertFalse(path.exists())
        self.assertEqual(self.client.get('/api/jobs/'+ident).status_code, 404)
        with self.client.session_transaction() as s:
            sid = s['sid']
        with self.state.lock:
            self.state.sessions[sid]['touched'] -= self.app.config['SESSION_TTL']+1
        self.state.cleanup()
        self.assertEqual(self.post('/api/generate', configuration()).status_code, 401)
        fresh = self.client.get('/api/catalog').get_json()
        self.assertNotEqual(fresh['csrf_token'], self.catalog['csrf_token'])

    def test_limits_and_failed_job_cleanup(self):
        self.assertEqual(self.post('/api/generate', configuration(count=51)).status_code, 400)
        self.assertEqual(self.post('/api/generate', configuration(seed=-1)).status_code, 400)
        self.assertEqual(self.post('/api/generate', configuration(datasets=['missing'])).status_code, 400)
        self.app.config['MAX_UPLOADS'] = 1
        data = dict(kind='colormap', filename='map.json', content=base64.b64encode(b'["blue","red"]').decode())
        self.assertEqual(self.post('/api/upload', data).status_code, 200)
        self.assertEqual(self.post('/api/upload', data).status_code, 429)
        self.app.config['MAX_TOTAL_ASSET_BYTES'] = 1
        self.assertEqual(self.post('/api/upload', data, client=self.other).status_code, 429)
        def fails(config, assets, output, *args, **kwargs):
            output.mkdir()
            (output/'partial.png').write_bytes(b'partial')
            raise TimeoutError('time limit')
        with patch('task_app.runtime.generate', fails):
            ident = self.post('/api/generate', configuration()).get_json()['id']
            result = self.wait_for(ident)
        self.assertEqual(result['status'], 'error')
        self.assertFalse((Path(self.folder.name)/ident).exists())
        self.assertEqual(self.client.get(f'/downloads/{ident}/everything.zip').status_code, 404)

    def test_large_npy_header_is_rejected_before_allocation(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, dict(descr='<f8', fortran_order=False, shape=(1000000,1000000)))
        with patch('task_app.assets.np.load', side_effect=AssertionError('must not allocate')):
            with self.assertRaisesRegex(ValueError, '3–128'):
                load_grid('hostile.npy', stream.getvalue())

    def test_selection_lists_are_bounded_and_typed(self):
        for selection in [['terrain_100']*129, [{}], 'terrain_100', ['x'*201]]:
            response = self.post('/api/generate', configuration(datasets=selection))
            self.assertEqual(response.status_code, 400, response.data)
        response = self.post('/api/compatibility', configuration(excluded_combinations=['x']*1001))
        self.assertEqual(response.status_code, 400)

    def test_generation_obeys_cancellation_during_sampling(self):
        calls = []
        def check():
            calls.append(True)
            if len(calls) == 3:
                raise TimeoutError('cancelled')
        with self.assertRaisesRegex(TimeoutError, 'cancelled'):
            generate(configuration(), self.state.base, Path(self.folder.name)/'cancelled', check=check)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
