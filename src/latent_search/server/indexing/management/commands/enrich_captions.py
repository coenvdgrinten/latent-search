"""
Management command to enrich photo captions using a Vision-Language Model.

Generates visual descriptions for photos and stores them in the database.
Supports batch processing with progress tracking and resume capability.
"""

import json
import logging
import time
from pathlib import Path
from typing import override

from django.core.management.base import BaseCommand, CommandParser

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.vlm import VLMService

logger = logging.getLogger(__name__)

# State file for resume capability
STATE_FILE = Path.home() / ".latent_search_vlm_state.json"


class Command(BaseCommand):
    help = "Enrich photo captions using a Vision-Language Model"

    @override
    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50,
            help="Number of images to process before saving progress checkpoint",
        )
        parser.add_argument(
            "--reprocess",
            action="store_true",
            help="Reprocess images that already have VLM captions",
        )
        parser.add_argument(
            "--clear-state",
            action="store_true",
            help="Clear previous processing state and start fresh",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be processed without actually running",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of images to process (for testing)",
        )

    def _load_state(self) -> dict:
        """Load processing state from disk."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_state(self, state: dict) -> None:
        """Save processing state to disk."""
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _clear_state(self) -> None:
        """Clear processing state file."""
        if STATE_FILE.exists():
            STATE_FILE.unlink()

    @override
    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        reprocess = options["reprocess"]
        clear_state = options["clear_state"]
        dry_run = options["dry_run"]
        limit = options["limit"]

        if clear_state:
            self._clear_state()
            self.stdout.write("Processing state cleared.")
            return

        # Load state
        state = self._load_state()
        processed_ids = set(state.get("processed_ids", []))

        # Determine queryset
        qs = IndexedMedia.objects.all()
        if not reprocess:
            qs = qs.filter(vlm_caption__exact="")

        # Exclude already processed IDs (resume support)
        if not reprocess and processed_ids:
            qs = qs.exclude(id__in=processed_ids)

        total = qs.count()
        if total == 0:
            self.stdout.write(
                self.style.WARNING(
                    "No images to process. "
                    "Use --reprocess to regenerate existing captions."
                )
            )
            return

        if limit:
            qs = qs[:limit]
            total = qs.count()

        self.stdout.write(f"\nFound {total} images to process.")
        if dry_run:
            self.stdout.write(
                self.style.WARNING("Dry run mode - no changes will be made.")
            )
            for media in qs[: min(10, total)]:
                self.stdout.write(f"  Would process: {media.filename}")
            if total > 10:
                self.stdout.write(f"  ... and {total - 10} more")
            return

        # Initialize VLM service
        self.stdout.write("\nInitializing VLM service...")
        vlm = VLMService()
        self.stdout.write(f"Model: {vlm.model_id}")
        self.stdout.write("Press Ctrl+C to pause (progress will be saved).\n")

        # Process in batches
        processed_count = 0
        error_count = 0
        start_time = time.time()

        try:
            for i, media in enumerate(qs):
                elapsed = time.time() - start_time
                eta_per_item = elapsed / (i + 1) if i > 0 else 0
                eta_remaining = eta_per_item * (total - i - 1)

                # Progress indicator
                pct = ((i + 1) / total) * 100
                self.stdout.write(
                    f"\r[{pct:5.1f}%] Processing {i+1}/{total} "
                    f"({eta_remaining/60:.1f}m remaining) ",
                    ending="",
                )

                try:
                    # Trigger lazy loading on first image
                    _ = vlm.model
                    _ = vlm.processor

                    caption = vlm.describe(media.file_path)

                    if caption:
                        media.vlm_caption = caption
                        media.save(update_fields=["vlm_caption"])
                        processed_ids.add(media.id)
                        processed_count += 1
                    else:
                        self.stderr.write(
                            f"\nWarning: Empty caption for {media.filename}"
                        )
                        error_count += 1

                except FileNotFoundError:
                    self.stderr.write(f"\nError: File not found: {media.file_path}")
                    error_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to process {media.file_path}: {e}", exc_info=True
                    )
                    self.stderr.write(f"\nError processing {media.filename}: {e}")
                    error_count += 1

                # Save checkpoint at batch boundaries
                if (i + 1) % batch_size == 0:
                    state["processed_ids"] = list(processed_ids)
                    self._save_state(state)

        except KeyboardInterrupt:
            self.stdout.write("\n\nInterrupted! Saving progress...")
            state["processed_ids"] = list(processed_ids)
            self._save_state(state)
            self.stdout.write(
                self.style.WARNING(
                    f"Paused at {processed_count}/{total}. " "Run again to resume."
                )
            )
            return

        # Final state save
        state["processed_ids"] = list(processed_ids)
        self._save_state(state)

        # Summary
        elapsed_total = time.time() - start_time
        self.stdout.write("\n\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("VLM Caption Enrichment Complete"))
        self.stdout.write("=" * 60)
        self.stdout.write(f"Processed: {processed_count} images")
        self.stdout.write(f"Errors: {error_count} images")
        self.stdout.write(f"Time: {elapsed_total/60:.1f} minutes")
        if processed_count > 0:
            self.stdout.write(f"Avg: {elapsed_total/processed_count:.1f}s per image")
        self.stdout.write("=" * 60)

        # Prompt for re-indexing
        self.stdout.write(
            "\nTip: Run 'python manage.py index_media <path>' "
            "to re-index with the enriched captions.\n"
        )
