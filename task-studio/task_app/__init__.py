"""Composable scalar-field task builder and web application."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache'))
