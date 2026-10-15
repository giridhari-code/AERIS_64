"""Kalofacts — standalone, versioned, dual-view interactive workspaces."""

from sentinel.kalofacts.config import KalofactsConfig, load_kalofacts_config
from sentinel.kalofacts.artifact import Kalofact, create_kalofact
from sentinel.kalofacts.store import (
    create_and_save,
    get,
    save,
    list_artifacts,
    delete,
    should_auto_open,
)
from sentinel.kalofacts.edit import (
    set_view_mode,
    replace_span,
    replace_all_content,
    restore_version,
    set_state,
)
from sentinel.kalofacts.export import export_source, create_share_token, resolve_share

__all__ = [
    "KalofactsConfig",
    "load_kalofacts_config",
    "Kalofact",
    "create_kalofact",
    "create_and_save",
    "get",
    "save",
    "list_artifacts",
    "delete",
    "should_auto_open",
    "set_view_mode",
    "replace_span",
    "replace_all_content",
    "restore_version",
    "set_state",
    "export_source",
    "create_share_token",
    "resolve_share",
]
