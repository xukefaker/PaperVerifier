__all__ = ["DeepChatService"]


def __getattr__(name: str):
    if name == "DeepChatService":
        from .service import DeepChatService

        return DeepChatService
    raise AttributeError(name)
