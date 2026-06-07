import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from exif import Image as ExifImage
from PIL import Image as PilImage

logger = logging.getLogger(__name__)


@dataclass
class MediaMetadata:
    width: int | None = None
    height: int | None = None
    taken_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None


def _dms_to_decimal(dms: tuple[float, float, float], ref: str) -> float:
    """Convert degrees/minutes/seconds + hemisphere ref to a decimal coordinate."""
    degrees, minutes, seconds = dms
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


class ExifService:
    def read_metadata(self, image_path: str | Path) -> MediaMetadata:
        """
        Extract width, height, capture time, and GPS coordinates from an image.
        Missing or unreadable fields are returned as None — never raises.
        """
        meta = MediaMetadata()

        # Width and height come from PIL — more reliable than EXIF tags
        try:
            with PilImage.open(image_path) as img:
                meta.width, meta.height = img.size
        except Exception as e:
            logger.warning(f"Could not read dimensions from {image_path}: {e}")

        # EXIF data for capture time and GPS
        try:
            with open(image_path, "rb") as f:
                exif = ExifImage(f)

            if not exif.has_exif:
                return meta

            # Capture time
            try:
                raw = exif.datetime_original
                meta.taken_at = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
            except Exception:
                pass

            # GPS coordinates
            try:
                meta.latitude = _dms_to_decimal(
                    exif.gps_latitude, exif.gps_latitude_ref
                )
                meta.longitude = _dms_to_decimal(
                    exif.gps_longitude, exif.gps_longitude_ref
                )
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"Could not read EXIF from {image_path}: {e}")

        return meta
