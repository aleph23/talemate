from urllib.parse import urljoin as _urljoin

__all__ = ["urljoin"]


def urljoin(base, *args):
    """Join a base URL with additional path components."""
    base = f"{base.rstrip('/')}/"
    return _urljoin(base, *args)
