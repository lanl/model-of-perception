"""Create a source ZIP for a new GitHub repository, excluding local state."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ['README.md', 'APP_GUIDE.md', 'DATASETS.md', 'DATA_SOURCES.md',
              'pyproject.toml', 'uv.lock', 'render.yaml', 'start.sh', 'gunicorn.conf.py',
              '.gitignore', '.python-version', '.env.example']
TREES = ['task_app', 'tests', 'scripts', '.github']
SKIP = {'__pycache__', '.cache', '.DS_Store', '.venv', 'runs'}


def package(output):
    paths = [ROOT/name for name in ROOT_FILES]
    for name in TREES:
        paths += [p for p in sorted((ROOT/name).rglob('*')) if p.is_file()
                  and not SKIP.intersection(p.relative_to(ROOT).parts)
                  and p.suffix not in ('.pyc', '.pyo')]
    paths = sorted(set(paths))
    output = output.resolve()
    if output in paths:
        raise ValueError('ZIP output must be outside the packaged source files.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for path in paths:
            assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()
    print(f'{output}: {len(paths)} files, {output.stat().st_size:,} bytes; verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/scalar-task-studio-github.zip')
    package(parser.parse_args().output)
