"""Local-first paper-to-LaTeX task packaging."""

from .task import MAX_PAGES, TaskError, create_task, load_status

__all__ = ["MAX_PAGES", "TaskError", "create_task", "load_status"]
__version__ = "0.1.0"
