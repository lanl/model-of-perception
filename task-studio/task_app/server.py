"""Public HTTP interface; production uses the supplied Gunicorn configuration."""
import argparse
import base64
import binascii
import hmac
import os
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit

from flask import Flask, abort, g, jsonify, request, send_file, session
from werkzeug.exceptions import HTTPException

from .contracts import OPTIONS, combinations, combination_id
from .parameter_space import summarize
from .runtime import Runtime, Busy, snapshot

HERE = Path(__file__).resolve().parent


def create_app(config=None):
    app = Flask(__name__, static_folder=None)
    public_url = os.environ.get('PUBLIC_BASE_URL') or os.environ.get('RENDER_EXTERNAL_URL')
    hosts = ['localhost', '127.0.0.1', '[::1]']
    if public_url:
        hosts.append(urlsplit(public_url).hostname)
    hosts.extend(v.strip() for v in os.environ.get('ALLOWED_HOSTS', '').split(',') if v.strip())
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY') or secrets.token_hex(32),
        TRUSTED_HOSTS=hosts, PUBLIC_BASE_URL=public_url,
        SESSION_COOKIE_NAME='scalar_task_session', SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax', SESSION_COOKIE_SECURE=bool(public_url and public_url.startswith('https://')),
        MAX_CONTENT_LENGTH=4_000_000,
        OUTPUT_DIR=os.environ.get('OUTPUT_DIR', 'runs'),
        MAX_TASKS=int(os.environ.get('MAX_TASKS', '50')),
        SESSION_TTL=int(os.environ.get('SESSION_TTL_SECONDS', '3600')),
        JOB_TTL=int(os.environ.get('JOB_TTL_SECONDS', '3600')),
        MAX_JOB_SECONDS=int(os.environ.get('MAX_JOB_SECONDS', '300')),
        MAX_SESSIONS=100, MAX_UPLOADS=20, MAX_UPLOAD_BYTES=20_000_000,
        MAX_TOTAL_ASSET_BYTES=64 * 1024 * 1024,
        MAX_JOBS_PER_SESSION=5, MAX_PENDING_JOBS=8,
        MAX_STORAGE_BYTES=int(os.environ.get('MAX_STORAGE_BYTES', '536870912')),
    )
    if config:
        app.config.update(config)
    if not 1 <= app.config['MAX_TASKS'] <= 200:
        raise ValueError('MAX_TASKS must be between 1 and 200.')
    for key in ('SESSION_TTL', 'JOB_TTL', 'MAX_JOB_SECONDS', 'MAX_SESSIONS', 'MAX_UPLOADS',
                'MAX_UPLOAD_BYTES', 'MAX_TOTAL_ASSET_BYTES', 'MAX_JOBS_PER_SESSION', 'MAX_PENDING_JOBS', 'MAX_STORAGE_BYTES'):
        if app.config[key] <= 0:
            raise ValueError(f'{key} must be positive.')
    state = Runtime(app.config, app.logger)
    app.extensions['task_runtime'] = state

    @app.before_request
    def protect():
        if request.path == '/healthz':
            return None
        if request.path in ('/', '/app.js', '/combinations-vis.js', '/style.css', '/favicon.ico'):
            return None
        sid = session.get('sid')
        if request.path == '/api/catalog' and request.method == 'GET':
            sid, visitor = state.get_session(sid, create=True)
            session['sid'] = sid
        else:
            sid, visitor = state.get_session(sid)
            if visitor is None:
                abort(401, description='Your session expired. Reload the page to start again.')
        g.sid, g.visitor = sid, visitor
        if request.method == 'POST':
            if not request.is_json:
                abort(415, description='Send an application/json request.')
            origin = request.headers.get('Origin')
            expected = app.config['PUBLIC_BASE_URL'] or request.host_url.rstrip('/')
            if origin and origin.rstrip('/') != expected.rstrip('/'):
                abort(403, description='Cross-site requests are not allowed.')
            if request.headers.get('Sec-Fetch-Site') == 'cross-site':
                abort(403, description='Cross-site requests are not allowed.')
            if not hmac.compare_digest(request.headers.get('X-CSRF-Token', ''), visitor['csrf']):
                abort(403, description='Session verification failed. Reload the page and try again.')

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.errorhandler(Busy)
    def busy_error(exc):
        return jsonify(error=str(exc)), 429, {'Retry-After': '10'}

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(TypeError)
    @app.errorhandler(UnicodeError)
    @app.errorhandler(binascii.Error)
    def input_error(exc):
        return jsonify(error=str(exc)), 400

    @app.get('/healthz')
    def health():
        return jsonify(status='ok')

    @app.get('/')
    @app.get('/<filename>')
    def static(filename='index.html'):
        if filename not in ('index.html', 'app.js', 'combinations-vis.js', 'style.css'):
            abort(404)
        return send_file(HERE / 'static' / filename)

    @app.get('/api/catalog')
    def catalog():
        with state.lock:
            return jsonify(g.visitor['assets'].catalog() | {
                'options': OPTIONS, 'csrf_token': g.visitor['csrf'],
                'limits': {'max_tasks': app.config['MAX_TASKS'], 'upload_bytes': 2_500_000,
                           'download_minutes': app.config['JOB_TTL'] // 60},
            })

    def payload():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object.')
        if request.path in ('/api/compatibility', '/api/generate'):
            for key in ('datasets', 'colormaps', *OPTIONS, 'excluded_combinations'):
                if key == 'excluded_combinations' and key not in data:
                    continue
                values = data.get(key)
                limit = 1000 if key == 'excluded_combinations' else 128
                if not isinstance(values, list) or len(values) > limit or any(not isinstance(v, str) or len(v) > 200 for v in values):
                    raise ValueError(f'Invalid selection list: {key}.')
        return data

    @app.post('/api/compatibility')
    def compatibility():
        data = payload()
        valid = combinations(data)
        available = combinations(data, include_excluded=True)
        with state.lock:
            space = summarize(valid, data, g.visitor['assets'])
        return jsonify(count=len(valid), combinations=valid, compatible_count=len(available),
                       combination_options=[dict(id=combination_id(stages), stages=stages,
                                                 included=stages in valid) for stages in available],
                       parameter_space=space)

    @app.post('/api/upload')
    def upload():
        data = payload()
        if not isinstance(data.get('content'), str) or not isinstance(data.get('filename'), str):
            raise ValueError('Provide a filename and base64-encoded content.')
        raw = base64.b64decode(data['content'], validate=True)
        if not 0 < len(raw) <= 2_500_000:
            raise ValueError('Upload limit: 2.5 MB.')
        name = Path(data['filename']).name
        with state.lock:
            visitor, assets = g.visitor, snapshot(g.visitor['assets'])
            if visitor['uploads'] >= app.config['MAX_UPLOADS'] or visitor['upload_bytes'] + len(raw) > app.config['MAX_UPLOAD_BYTES']:
                raise Busy('This session has reached its upload limit.')
            if data.get('kind') == 'dataset':
                ident = assets.add_grid(name, raw, name, [float(v) for v in data.get('extent', [0, 1])])
            elif data.get('kind') == 'colormap':
                ident = assets.add_map(name, raw, name)
            else:
                raise ValueError('Unknown upload kind.')
            memory = sum(v['values'].nbytes for k, v in assets.datasets.items() if k not in state.base.datasets)
            memory += sum(len(v['colors']) * 256 for k, v in assets.maps.items() if k not in state.base.maps)
            other_memory = sum(v['asset_bytes'] for v in state.sessions.values() if v is not visitor)
            if other_memory + memory > app.config['MAX_TOTAL_ASSET_BYTES']:
                raise Busy('The demo has reached its upload memory limit. Please try again later.')
            visitor['assets'], visitor['asset_bytes'] = assets, memory
            visitor['uploads'] += 1
            visitor['upload_bytes'] += len(raw)
            return jsonify({'id': ident, **assets.catalog()})

    @app.post('/api/generate')
    def generate():
        return jsonify(id=state.submit(g.sid, payload())), 202

    @app.get('/api/jobs/<ident>')
    def job(ident):
        result = state.status(g.sid, ident)
        if result is None:
            abort(404, description='Unknown or expired batch.')
        return jsonify(result)

    @app.get('/downloads/<ident>/<path:filename>')
    def download(ident, filename):
        if not re.fullmatch(r'(tasks\.(csv|md|json)|manifest\.json|images\.zip|everything\.zip|images/task_\d{4}\.png)', filename):
            abort(404)
        path = state.download(g.sid, ident, filename)
        if path is None:
            abort(404, description='Unknown or expired download.')
        try:
            return send_file(path, as_attachment=not filename.endswith('.png'), download_name=path.name)
        except FileNotFoundError:
            abort(404, description='This download has expired.')

    return app


def main():
    parser = argparse.ArgumentParser(description='Scalar Task Studio local preview')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8765')))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = {'OUTPUT_DIR': args.output} if args.output else None
    app = create_app(config)
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        app.extensions['task_runtime'].close()
