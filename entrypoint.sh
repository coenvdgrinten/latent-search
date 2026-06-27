#!/bin/sh
set -e

# PUID/PGID entrypoint — runs as root, adjusts the container user to match
# env vars, fixes ownership, then drops privileges via gosu.

PUID="${PUID:-1000}"
PGID="${PGID:-$PUID}"

echo "entrypoint: uid=$PUID gid=$PGID"

# Update existing group/user or create if missing
groupmod -o -g "$PGID" app 2>/dev/null || addgroup -g "$PGID" app
usermod -o -u "$PUID" -g "$PGID" appuser 2>/dev/null || adduser -u "$PUID" -g "$PGID" -D -h /app appuser

# Fix ownership so the new UID can write
chown -R "$PUID:$PGID" /app /cache 2>/dev/null || true

exec gosu "$PUID:$PGID" "$@"
