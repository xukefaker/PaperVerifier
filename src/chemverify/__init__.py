"""ChemVerify application package."""

__all__ = ["SearchEngine", "Settings"]


def __getattr__(name: str):
    if name == "Settings":
        from .config import Settings

        return Settings
    if name == "SearchEngine":
        from .search import SearchEngine

        return SearchEngine
    raise AttributeError(name)
