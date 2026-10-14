"""AjeebWork / autonomous workspace capabilities for SENTINEL."""

from neurofield.ajeebwork.config import AjeebWorkConfig, load_ajeebwork_config
from neurofield.ajeebwork.workspace_files import (
    read_file,
    write_file,
    edit_file,
    delete_file,
    batch_process,
)
from neurofield.ajeebwork.plan_execute import execute_plan, plan_execute_report, build_plan
from neurofield.ajeebwork.subagents import run_subagents, subagents_report
from neurofield.ajeebwork.tasks import create_task, get_task, list_tasks, tasks_report, run_task_inline
from neurofield.ajeebwork.shared_memory import memory_get, memory_set, memory_report, memory_append_transcript
from neurofield.ajeebwork.documents import documents_report
from neurofield.ajeebwork.draft_edit import edit_span, preview_span
from neurofield.ajeebwork.skills import save_skill, load_skill, list_skills, register_connector, call_connector

__all__ = [
    "AjeebWorkConfig",
    "load_ajeebwork_config",
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
