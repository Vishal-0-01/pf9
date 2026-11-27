"""Vercel entrypoint (Flask). The application lives in backend/; this shim only exposes `app`."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from webapp import app  # noqa: E402,F401


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
