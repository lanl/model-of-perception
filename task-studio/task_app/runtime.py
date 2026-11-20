"""Bounded, single-process generation queue and isolated visitor state."""
import copy
import json
from pathlib import Path
import queue
import re
import secrets
import shutil
import threading
import time

from .assets import Assets
from .contracts import combinations
from .export import generate


class Busy(Exception):
    pass


def snapshot(assets):
    result = copy.copy(assets)
    result.datasets = assets.datasets.copy()
    result.maps = assets.maps.copy()
    result.dataset_warnings = assets.dataset_warnings.copy()
    return result


class Runtime:
    def __init__(self, config, logger):
        self.config, self.logger = config, logger
        self.root = Path(config['OUTPUT_DIR']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base = Assets()
        self.sessions, self.jobs = {}, {}
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=config['MAX_PENDING_JOBS'])
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True, name='task-generation')
        self.thread.start()

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)

    def cleanup(self):
        with self.lock:
            now = time.time()
            for ident, job in list(self.jobs.items()):
                if job['status'] not in ('queued', 'running') and now - job['finished'] > self.config['JOB_TTL']:
                    shutil.rmtree(self.root / ident, ignore_errors=True)
                    del self.jobs[ident]
            active = {job['owner'] for job in self.jobs.values() if job['status'] in ('queued', 'running')}
            for sid, visitor in list(self.sessions.items()):
                if sid not in active and now - visitor['touched'] > self.config['SESSION_TTL']:
                    del self.sessions[sid]
                    for ident, job in list(self.jobs.items()):
                        if job['owner'] == sid:
                            shutil.rmtree(self.root / ident, ignore_errors=True)
                            del self.jobs[ident]
            # Expire files left by previous processes, never arbitrary directories.
            for path in self.root.iterdir():
                if path.is_dir() and re.fullmatch(r'[a-f0-9]{32}', path.name) and path.name not in self.jobs:
                    if now - path.stat().st_mtime > self.config['JOB_TTL']:
                        shutil.rmtree(path, ignore_errors=True)

    def get_session(self, sid, create=False):
        self.cleanup()
        with self.lock:
            visitor = self.sessions.get(sid) if isinstance(sid, str) else None
            if visitor is None and create:
                if len(self.sessions) >= self.config['MAX_SESSIONS']:
                    raise Busy('The demo is busy. Please try again later.')
                sid = secrets.token_hex(32)
                visitor = dict(assets=snapshot(self.base), csrf=secrets.token_hex(32),
                               uploads=0, upload_bytes=0, asset_bytes=0, touched=time.time())
                self.sessions[sid] = visitor
            if visitor is not None:
                visitor['touched'] = time.time()
            return sid, visitor

    def disk_usage(self):
        total = 0
        for path in self.root.rglob('*'):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
        return total

    def submit(self, sid, config):
        with self.lock:
            assets = snapshot(self.sessions[sid]['assets'])
            if type(config.get('count')) is not int or not 1 <= config['count'] <= self.config['MAX_TASKS']:
                raise ValueError(f"Task count must be 1–{self.config['MAX_TASKS']}.")
            if type(config.get('seed')) is not int or not 0 <= config['seed'] < 2**53:
                raise ValueError('Seed must be an integer from 0 to 2^53−1.')
            if not combinations(config):
                raise ValueError('No included compatible combinations. Include at least one combination.')
            for key, source in [('datasets', assets.datasets), ('colormaps', assets.maps)]:
                if not isinstance(config.get(key), list) or not config[key] or any(not isinstance(v, str) or v not in source for v in config[key]):
                    raise ValueError(f'Select valid {key}.')
            owned = [j for j in self.jobs.values() if j['owner'] == sid]
            if any(j['status'] in ('running', 'queued') for j in owned):
                raise Busy('You already have a batch queued or running.')
            if self.queue.full():
                raise Busy('The generation queue is full. Please try again shortly.')
            if self.disk_usage() >= self.config['MAX_STORAGE_BYTES']:
                raise Busy('Download storage is full. Please try again after older batches expire.')
            if len(owned) >= self.config['MAX_JOBS_PER_SESSION']:
                old = min((k for k, j in self.jobs.items() if j['owner'] == sid), key=lambda k: self.jobs[k]['finished'])
                shutil.rmtree(self.root / old, ignore_errors=True)
                del self.jobs[old]
            ident = secrets.token_hex(16)
            self.jobs[ident] = dict(owner=sid, status='queued', completed=0, total=config['count'])
            self.queue.put_nowait((ident, config, assets))
            return ident

    def work(self):
        while not self.stopped.is_set():
            try:
                ident, config, assets = self.queue.get(timeout=1)
            except queue.Empty:
                self.cleanup()
                continue
            with self.lock:
                self.jobs[ident]['status'] = 'running'
            deadline = time.monotonic() + self.config['MAX_JOB_SECONDS']

            def check():
                if self.stopped.is_set() or time.monotonic() > deadline:
                    raise TimeoutError('This batch exceeded the generation time limit. Try fewer tasks or simpler combinations.')

            def progress(done, total):
                check()
                with self.lock:
                    self.jobs[ident].update(completed=done, total=total)
                    if self.disk_usage() > self.config['MAX_STORAGE_BYTES']:
                        raise RuntimeError('Download storage limit reached. Please try a smaller batch later.')

            try:
                generate(config, assets, self.root / ident, progress, check=check)
                with self.lock:
                    if self.disk_usage() > self.config['MAX_STORAGE_BYTES']:
                        raise RuntimeError('Download storage limit reached. Please try a smaller batch later.')
                    self.jobs[ident].update(status='complete', finished=time.time())
            except Exception as exc:
                self.logger.exception('Generation failed for batch %s', ident)
                shutil.rmtree(self.root / ident, ignore_errors=True)
                with self.lock:
                    message = str(exc) if isinstance(exc, (ValueError, TimeoutError, RuntimeError)) else 'Generation failed. Please try another selection.'
                    self.jobs[ident].update(status='error', error=message, finished=time.time())
            finally:
                del assets, config
                self.queue.task_done()
                self.cleanup()

    def status(self, sid, ident):
        with self.lock:
            job = self.jobs.get(ident)
            if job is None or job['owner'] != sid:
                return None
            result = {k: v for k, v in job.items() if k not in ('owner', 'finished')}
            if job['status'] == 'complete':
                result.update(records=json.loads((self.root / ident / 'tasks.json').read_text()),
                              manifest=json.loads((self.root / ident / 'manifest.json').read_text()))
            return result

    def download(self, sid, ident, filename):
        with self.lock:
            job = self.jobs.get(ident)
            if job is None or job['owner'] != sid or job['status'] != 'complete':
                return None
            path = self.root / ident / filename
            return path if path.is_file() else None
