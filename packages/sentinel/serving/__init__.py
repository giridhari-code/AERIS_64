"""SENTINEL serving package. Import create_app from sentinel.serving.server when FastAPI is installed."""

__all__ = ["create_app"]

def __getattr__(name: str):
    if name == "create_app":
        from sentinel.serving.server import create_app
        return create_app
    raise AttributeError(name)

