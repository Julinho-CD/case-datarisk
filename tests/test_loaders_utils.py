from pathlib import Path

from app.loaders import resolve_project_path
from src.config import PROJECT_ROOT


def test_resolve_project_path_keeps_absolute_path():
    absolute = PROJECT_ROOT / "README.md"
    out = resolve_project_path(absolute)
    assert out == absolute


def test_resolve_project_path_expands_relative_path_from_project_root():
    out = resolve_project_path(Path("app") / "streamlit_app.py")
    assert out == PROJECT_ROOT / "app" / "streamlit_app.py"
