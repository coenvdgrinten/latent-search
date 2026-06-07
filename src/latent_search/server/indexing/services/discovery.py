from collections.abc import Iterator
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DiscoveryService:
    def discover_media(self, root_path: str | Path) -> Iterator[Path]:
        """
        Recursively discover supported media files in the given root path.
        """
        root = Path(root_path)
        if not root.is_dir():
            raise NotADirectoryError(f"{root_path} is not a directory")

        for path in root.rglob("*"):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path
