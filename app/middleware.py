from __future__ import annotations

from typing import Callable, Optional


class PrefixMiddleware:
    """
    Allow the app to run behind a URL prefix *without* requiring Flask to mount routes under that prefix.

    This supports two common reverse-proxy setups:
    - Proxy preserves prefix upstream: client requests `/thingid/...`, upstream also sees `/thingid/...`
      -> middleware strips `/thingid` from PATH_INFO and sets SCRIPT_NAME.
    - Proxy strips prefix upstream but forwards `X-Forwarded-Prefix: /thingid`
      -> ProxyFix handles SCRIPT_NAME; this middleware does nothing.
    """

    def __init__(self, app: Callable, prefix: Optional[str]) -> None:
        self.app = app
        self.prefix = (prefix or "").rstrip("/")

    def __call__(self, environ, start_response):
        prefix = self.prefix
        if not prefix or prefix == "/":
            return self.app(environ, start_response)

        path = environ.get("PATH_INFO", "") or ""
        if not path.startswith(prefix + "/") and path != prefix:
            return self.app(environ, start_response)

        # If a reverse proxy already set SCRIPT_NAME (e.g. via ProxyFix + X-Forwarded-Prefix),
        # don't try to second-guess it.
        if environ.get("SCRIPT_NAME"):
            return self.app(environ, start_response)

        environ["SCRIPT_NAME"] = prefix
        environ["PATH_INFO"] = path[len(prefix) :] or "/"
        return self.app(environ, start_response)

