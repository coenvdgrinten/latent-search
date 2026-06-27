"""Create a user from environment variables (LS_USERNAME / LS_PASSWORD).

Usage:
    ./manage create_user

If the user already exists, the password is updated.
"""

import logging
import os
from typing import override

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Create or update a user from LS_USERNAME and LS_PASSWORD env vars."

    @override
    def handle(self, *args, **options):
        username = os.environ.get("LS_USERNAME", "").strip()
        password = os.environ.get("LS_PASSWORD", "").strip()

        if not username or not password:
            self.stderr.write("Set LS_USERNAME and LS_PASSWORD environment variables.")
            return

        user, created = User.objects.get_or_create(username=username)
        user.set_password(password)
        user.is_staff = True
        user.save()

        if created:
            self.stdout.write(self.style.SUCCESS(f"Created user '{username}'"))
        else:
            self.stdout.write(self.style.SUCCESS(f"Updated password for '{username}'"))
