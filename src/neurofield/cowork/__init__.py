"""Cowork / autonomous workspace capabilities for NeuroField."""

from neurofield.cowork.config import CoworkConfig, load_cowork_config
from neurofield.cowork.workspace_files import (
    read_file,
    write_file,
    edit_file,
    delete_file,
    batch_process,
)
from neurofield.cowork.plan_execute import execute_plan, plan_execute_report, build_plan
from neurofield.cowork.subagents import run_subagents, subagents_report
from neurofield.cowork.tasks import create_task, get_task, list_tasks, tasks_report, run_task_inline
from neurofield.cowork.shared_memory import memory_get, memory_set, memory_report, memory_append_transcript
from neurofield.cowork.documents import documents_report
from neurofield.cowork.draft_edit import edit_span, preview_span
from neurofield.cowork.skills import save_skill, load_skill, list_skills, register_connector, call_connector

__all__ = [
    "CoworkConfig",
    "load_cowork_config",
    "read_file",
    "write_file",
    "edit_file",
    "delete_file",
    "batch_process",
    "execute_plan",
    "plan_execute_report",
    "build_plan",
    "run_subagents",
    "subagents_report",
    "create_task",
    "get_task",
    "list_tasks",
    "tasks_report",
    "run_task_inline",
    "memory_get",
    "memory_set",
    "memory_report",
    "memory_append_transcript",
    "documents_report",
    "edit_span",
    "preview_span",
    "save_skill",
    "load_skill",
    "list_skills",
    "register_connector",
    "call_connector",
]
