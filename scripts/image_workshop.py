"""Forge extension entry point; no monkey patches or package installation."""

import sys
from pathlib import Path

extension_root = str(Path(__file__).resolve().parents[1])
if extension_root not in sys.path:
    sys.path.insert(0, extension_root)

from modules import script_callbacks
from forge_image_workshop.ui import create_ui

script_callbacks.on_ui_tabs(create_ui)
