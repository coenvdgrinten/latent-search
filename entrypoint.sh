#!/bin/sh
set -e

# ----------------------------------------------------------------------
# PUID / PGID entrypoint — adjusts the 'appuser' UID/GID at runtime so
# container permissions match the host's file ownership.
#
# Usage in docker-compose.yml:
#   environment:
#     - PUID=99       # defaults to current appuser UID if unset
#     - PGID=100      # defaults to PUID if unset
#
# The entrypoint runs as root, adjusts the user/group, fixes ownership
# of writable directories, then drops privileges with gosu.
# ----------------------------------------------------------------------

if [ -n "$PUID" ] && [ "$PUID" != "0" ]; then
    PGID_VAL="${PGID:-$PUID}"

    echo "entrypoint: setting appuser uid=$PUID gid=$PGID_VAL"

    # Update group first (if it already exists, -o allows non-unique)
    groupmod -o -g "$PGID_VAL" app 2>/dev/null || true

    # Update user
    usermod -o -u "$PUID" -g "$PGID_VAL" appuser 2>/dev/null || true

    # Fix ownership of writable directories so the new UID can write
    chown -R appuser:app /app/data /app/media /cache 2>/dev/null || true
fi

exec gosu appuser:app "$@"
