#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys


def main():
    """Run administrative tasks."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "latent_search.server.config.settings"
    )

    # If 'test' is run without arguments, default to testing the indexing app
    # (since Django discovery doesn't find it automatically due to the nested structure)
    if len(sys.argv) == 2 and sys.argv[1] == "test":
        sys.argv.append("latent_search.server.indexing")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
