from langfuse import get_client

langfuse = get_client()


def flush_langfuse() -> None:
    """Ensure pending Langfuse events are sent."""
    langfuse.flush()
