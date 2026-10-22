"""Extract image metadata and any explicitly recorded location for reef imagery.

The module does not infer a location from visual content. It looks only at
embedded EXIF/XMP metadata and image-associated label/metadata files.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from PIL import ExifTags, Image

try:
    import yaml
except ImportError:  # pragma: no cover - project requirements include PyYAML
    yaml = None


_LOCATION_KEYS = {
    "location", "locationname", "placename", "place", "site", "sitename",
    "island", "islandname", "atoll", "atollname", "reef", "reefname",
    "country", "countryname", "region", "state", "province", "provincestate",
    "city", "locality", "sublocation", "locationdescription", "gpsplacename",
}
_LAT_KEYS = {"latitude", "lat", "gpslatitude", "gpslat"}
_LON_KEYS = {"longitude", "lon", "lng", "long", "gpslongitude", "gpslon", "gpslng"}
_FILENAME_KEYS = {"filename", "file", "image", "imagefile", "imagefilename", "imagepath", "filepath", "photofile", "photo", "photoid", "imageid", "imageidentifier", "photoidentifier", "video", "videofile", "videofilename", "videopath", "videoid", "surveyvideo", "recording", "recordingname", "clip", "clipid"}
_MANIFEST_NAMES = {
    "metadata.csv", "image_metadata.csv", "photo_metadata.csv", "survey_metadata.csv",
    "labels.csv", "annotations.csv", "images.csv", "metadata.json", "image_metadata.json",
    "labels.json", "annotations.json", "metadata.yaml", "metadata.yml", "labels.yaml",
    "labels.yml", "annotations.yaml", "annotations.yml",
}


def _normalise_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    # Decimal coordinates and common N/S/E/W suffixes are supported here.
    match = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*([NSEWnsew]?)\s*", text)
    if not match:
        return None
    result = float(match.group(1))
    if match.group(2).upper() in {"S", "W"}:
        result = -abs(result)
    elif match.group(2).upper() in {"N", "E"}:
        result = abs(result)
    return result


def _rational(value: Any) -> float | None:
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            denominator = float(value.denominator)
            return float(value.numerator) / denominator if denominator else None
        if isinstance(value, (tuple, list)) and len(value) == 2:
            denominator = float(value[1])
            return float(value[0]) / denominator if denominator else None
        return _number(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _dms_to_decimal(value: Any, ref: Any = None) -> float | None:
    if value is None:
        return None
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", errors="ignore")
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        parts = [_rational(part) for part in value[:3]]
        if all(part is not None for part in parts):
            result = float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
        else:
            return None
    else:
        result = _number(value)
    if result is None:
        return None
    direction = str(ref or "").strip().upper()
    if direction in {"S", "W", "2", "W\x00", "S\x00"}:
        result = -abs(result)
    elif direction in {"N", "E", "1", "N\x00", "E\x00"}:
        result = abs(result)
    return result


def _is_valid_pair(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def _flatten_values(record: Any) -> list[tuple[str, Any]]:
    """Flatten nested dicts while preserving every key's semantic name."""
    items: list[tuple[str, Any]] = []
    if isinstance(record, dict):
        for key, value in record.items():
            items.append((_normalise_key(key), value))
            if isinstance(value, dict):
                items.extend(_flatten_values(value))
    elif isinstance(record, list):
        for value in record:
            if isinstance(value, dict):
                items.extend(_flatten_values(value))
    return items


def _clean_location_name(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown", "n/a", "na", "not available", "nan"}:
        return None
    return text


def _location_from_record(record: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(record, (dict, list)):
        return None

    flat = _flatten_values(record)
    # Name fields are deliberately explicit: arbitrary captions or filenames are
    # not treated as location metadata.
    names: list[str] = []
    lat = lon = None
    lat_ref = lon_ref = None
    for key, value in flat:
        if key in _LOCATION_KEYS:
            name = _clean_location_name(value)
            if name and name not in names:
                names.append(name)
        elif key in _LAT_KEYS and lat is None:
            lat = _dms_to_decimal(value)
        elif key in _LON_KEYS and lon is None:
            lon = _dms_to_decimal(value)
        elif key in {"gpslatituderef", "latituderef"}:
            lat_ref = value
        elif key in {"gpslongituderef", "longituderef"}:
            lon_ref = value

    # Apply optional direction/ref fields for sidecar coordinates.
    if lat is not None and str(lat_ref or "").strip().upper() in {"S", "2"}:
        lat = -abs(lat)
    if lon is not None and str(lon_ref or "").strip().upper() in {"W", "2"}:
        lon = -abs(lon)

    # A few common geospatial sidecars store GeoJSON-style coordinates as [lon, lat].
    if not _is_valid_pair(lat, lon):
        for key, value in flat:
            if key in {"coordinates", "gpscoordinates", "coordinatepair"} and isinstance(value, (list, tuple)) and len(value) >= 2:
                candidate_lon, candidate_lat = _number(value[0]), _number(value[1])
                if _is_valid_pair(candidate_lat, candidate_lon):
                    lat, lon = candidate_lat, candidate_lon
                    break

    valid_pair = _is_valid_pair(lat, lon)
    if not names and not valid_pair:
        return None

    # Prefer specific named details first (site/island/atoll) if they were found.
    # Deduplicate case-insensitively while preserving order.
    unique_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name.casefold() not in seen:
            unique_names.append(name)
            seen.add(name.casefold())
    display = ", ".join(unique_names)
    if not display and valid_pair:
        display = f"{lat:.6f}, {lon:.6f}"
    return {
        "available": True,
        "display": display,
        "name": display or None,
        "latitude": round(lat, 7) if valid_pair else None,
        "longitude": round(lon, 7) if valid_pair else None,
        "source": source,
    }


def _location_from_exif(image: Image.Image) -> dict[str, Any] | None:
    try:
        exif = image.getexif()
        if not exif:
            return None
        gps: Any = None
        try:
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except (AttributeError, KeyError, TypeError):
            gps = exif.get(34853)
        if not isinstance(gps, dict) or not gps:
            return None
        gps_tags = {ExifTags.GPSTAGS.get(tag, str(tag)): value for tag, value in gps.items()}
        lat = _dms_to_decimal(gps_tags.get("GPSLatitude"), gps_tags.get("GPSLatitudeRef"))
        lon = _dms_to_decimal(gps_tags.get("GPSLongitude"), gps_tags.get("GPSLongitudeRef"))
        if not _is_valid_pair(lat, lon):
            return None
        return {
            "available": True,
            "display": f"{lat:.6f}, {lon:.6f}",
            "name": None,
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "source": "Embedded EXIF GPS",
        }
    except Exception:
        return None


def _xmp_fields(raw_xmp: Any) -> dict[str, str]:
    if isinstance(raw_xmp, bytes):
        raw_xmp = raw_xmp.decode("utf-8", errors="ignore")
    if not isinstance(raw_xmp, str) or not raw_xmp.strip():
        return {}
    fields: dict[str, str] = {}
    try:
        root = ET.fromstring(raw_xmp)
        for element in root.iter():
            local = element.tag.rsplit("}", 1)[-1]
            if element.text and element.text.strip():
                fields[local] = element.text.strip()
            for key, value in element.attrib.items():
                fields[key.rsplit("}", 1)[-1]] = value
    except ET.ParseError:
        # Some files keep an incomplete XMP packet; only explicit field names count.
        for key in (*_LOCATION_KEYS, *_LAT_KEYS, *_LON_KEYS):
            match = re.search(rf"(?:[\w.-]+:)?{re.escape(key)}\s*=\s*['\"]([^'\"]+)['\"]", raw_xmp, re.I)
            if match:
                fields[key] = match.group(1)
    return fields


def _location_from_xmp(image: Image.Image) -> dict[str, Any] | None:
    chunks: list[Any] = []
    for key in ("xmp", "XML:com.adobe.xmp", "Raw profile type xmp"):
        if key in image.info:
            chunks.append(image.info[key])
    try:
        exif = image.getexif()
        if exif and exif.get(700):  # EXIF XMLPacket / XMP
            chunks.append(exif.get(700))
    except Exception:
        pass
    for raw in chunks:
        result = _location_from_record(_xmp_fields(raw), "Embedded XMP metadata")
        if result:
            return result
    return None


def _row_matches_image(record: dict[str, Any], image_path: Path) -> bool:
    for key, value in record.items():
        if _normalise_key(key) in _FILENAME_KEYS:
            candidate = str(value or "").replace("\\", "/").strip()
            if candidate and (Path(candidate).name.casefold() == image_path.name.casefold()
                              or Path(candidate).stem.casefold() == image_path.stem.casefold()):
                return True
    return False


def _xml_to_dict(path: Path) -> dict[str, Any] | None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None

    def convert(node: ET.Element) -> Any:
        children = list(node)
        if not children:
            return (node.text or "").strip()
        result: dict[str, Any] = {}
        for child in children:
            key = child.tag.rsplit("}", 1)[-1]
            value = convert(child)
            if key in result:
                if not isinstance(result[key], list):
                    result[key] = [result[key]]
                result[key].append(value)
            else:
                result[key] = value
        return result
    value = convert(root)
    return value if isinstance(value, dict) else None


def _load_metadata_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8-sig"))
        if suffix in {".yaml", ".yml"} and yaml is not None:
            return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        if suffix == ".xml":
            return _xml_to_dict(path)
        if suffix == ".txt":
            content = path.read_text(encoding="utf-8-sig", errors="ignore")
            record: dict[str, str] = {}
            for line in content.splitlines():
                match = re.match(r"\s*([A-Za-z][A-Za-z _-]{1,40})\s*[:=]\s*(.*?)\s*$", line)
                if match and _normalise_key(match.group(1)) in (_LOCATION_KEYS | _LAT_KEYS | _LON_KEYS | {"gpslatituderef", "gpslongituderef"}):
                    record[match.group(1)] = match.group(2)
            return record
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def _location_from_metadata_file(path: Path, image_path: Path, *, explicitly_selected: bool = False) -> dict[str, Any] | None:
    data = _load_metadata_file(path)
    if data is None:
        return None
    records: list[Any] = []
    if isinstance(data, list):
        matching = [row for row in data if isinstance(row, dict) and _row_matches_image(row, image_path)]
        if matching:
            records.extend(matching)
        elif explicitly_selected and len(data) == 1:
            records.extend(data)
    elif isinstance(data, dict):
        # Common JSON format: {"IMG_001.jpg": {"location": ...}}
        for key, value in data.items():
            if isinstance(value, (dict, list)) and (str(key).casefold() in {image_path.name.casefold(), image_path.stem.casefold()}):
                records.append(value)
        if not records:
            # If the file is explicitly supplied for this image, or it clearly
            # looks like a single record, it may be used directly.
            has_record_fields = any(_normalise_key(k) in (_LOCATION_KEYS | _LAT_KEYS | _LON_KEYS | _FILENAME_KEYS) for k in data)
            if explicitly_selected and has_record_fields:
                if not any(_normalise_key(k) in _FILENAME_KEYS for k in data) or _row_matches_image(data, image_path):
                    records.append(data)
            elif any(_normalise_key(k) in _FILENAME_KEYS for k in data) and _row_matches_image(data, image_path):
                records.append(data)
    for record in records:
        result = _location_from_record(record, f"Label/metadata file: {path.name}")
        if result:
            return result
    return None


def _is_image_sidecar(metadata_path: Path, image_path: Path) -> bool:
    """Return True when the filename itself associates the sidecar with this image."""
    suffixes = (".json", ".yaml", ".yml", ".xml", ".csv", ".txt")
    expected = {image_path.name.casefold() + suffix for suffix in suffixes}
    expected.update({image_path.stem.casefold() + suffix for suffix in suffixes})
    try:
        return metadata_path.parent.resolve() == image_path.parent.resolve() and metadata_path.name.casefold() in expected
    except OSError:
        return metadata_path.parent == image_path.parent and metadata_path.name.casefold() in expected


def _candidate_metadata_files(image_path: Path, search_roots: Iterable[str | Path]) -> list[Path]:
    suffixes = (".json", ".yaml", ".yml", ".xml", ".csv", ".txt")
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend((image_path.with_suffix(image_path.suffix + suffix), image_path.with_suffix(suffix)))
    for root_value in search_roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        # Look for a small, fixed set of conventional manifests only; avoid a
        # recursive crawl through large image datasets during every inference.
        roots = [root, root / "metadata", root / "labels", root / "annotations"]
        for folder in roots:
            if folder.is_dir():
                for filename in _MANIFEST_NAMES:
                    candidate = folder / filename
                    if candidate.is_file():
                        candidates.append(candidate)
                # Some coral survey datasets store per-photo location details in
                # named CSVs (for example, *_corals_dryad.csv) rather than a generic
                # metadata.csv. Check shallow CSV manifests and match by Photo_ID.
                for candidate in folder.glob("*.csv"):
                    try:
                        if candidate.is_file() and candidate.stat().st_size <= 20 * 1024 * 1024:
                            candidates.append(candidate)
                    except OSError:
                        continue
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.is_file():
            resolved = str(candidate.resolve())
            if resolved not in seen:
                unique.append(candidate)
                seen.add(resolved)
    return unique


def extract_image_metadata(
    image_path: str | Path,
    label_path: str | Path | None = None,
    search_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Return image basics and location found in embedded or associated metadata.

    Location is shown only when a named place is explicitly recorded or a valid
    latitude/longitude pair is present. This function never guesses from pixels.
    """
    path = Path(image_path)
    info: dict[str, Any] = {
        "filename": path.name,
        "format": None,
        "width": None,
        "height": None,
        "captured_at": None,
        "location": {
            "available": False,
            "display": "No location available",
            "name": None,
            "latitude": None,
            "longitude": None,
            "source": None,
        },
    }
    candidates: list[dict[str, Any]] = []
    try:
        with Image.open(path) as image:
            info["format"] = image.format
            info["width"], info["height"] = image.size
            exif = image.getexif()
            date_value = (exif.get(36867) or exif.get(306)) if exif else None  # DateTimeOriginal / DateTime
            if date_value:
                info["captured_at"] = str(date_value)
            exif_location = _location_from_exif(image)
            if exif_location:
                candidates.append(exif_location)
            xmp_location = _location_from_xmp(image)
            if xmp_location:
                candidates.append(xmp_location)
    except (OSError, ValueError, SyntaxError):
        # The model pipeline already handles unsupported/corrupt image files;
        # metadata extraction should fail safely and show no location.
        pass

    paths_to_check: list[tuple[Path, bool]] = []
    if label_path:
        explicit_path = Path(label_path)
        if explicit_path.is_file():
            paths_to_check.append((explicit_path, True))
    paths_to_check.extend((item, False) for item in _candidate_metadata_files(path, search_roots))
    seen: set[str] = set()
    for metadata_path, explicit in paths_to_check:
        key = str(metadata_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        result = _location_from_metadata_file(
            metadata_path,
            path,
            explicitly_selected=explicit or _is_image_sidecar(metadata_path, path),
        )
        if result:
            candidates.append(result)

    # Prefer a label's human-readable place name, while keeping the most precise
    # coordinate pair available from EXIF/XMP/label metadata.
    name_candidate = next((c for c in candidates if c.get("name")), None)
    coordinate_candidate = next((c for c in candidates if _is_valid_pair(c.get("latitude"), c.get("longitude"))), None)
    if name_candidate or coordinate_candidate:
        name = name_candidate.get("name") if name_candidate else None
        lat = coordinate_candidate.get("latitude") if coordinate_candidate else None
        lon = coordinate_candidate.get("longitude") if coordinate_candidate else None
        display = name or (f"{lat:.6f}, {lon:.6f}" if coordinate_candidate else "No location available")
        sources = []
        for candidate in candidates:
            src = candidate.get("source")
            if src and ((candidate.get("name") and name) or _is_valid_pair(candidate.get("latitude"), candidate.get("longitude"))):
                if src not in sources:
                    sources.append(src)
        info["location"] = {
            "available": True,
            "display": display,
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "source": "; ".join(sources) or "Image metadata",
        }
    return info


_ISO6709_RE = re.compile(
    r"^\s*([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?(?:CRS[^/]*)?/\s*$",
    re.IGNORECASE,
)


def _location_from_iso6709(value: Any, source: str) -> dict[str, Any] | None:
    """Parse an ISO 6709 coordinate string used by some MP4/MOV metadata."""
    if value is None:
        return None
    match = _ISO6709_RE.match(str(value))
    if not match:
        return None
    lat, lon = _number(match.group(1)), _number(match.group(2))
    if not _is_valid_pair(lat, lon):
        return None
    return {
        "available": True,
        "display": f"{lat:.6f}, {lon:.6f}",
        "name": None,
        "latitude": round(lat, 7),
        "longitude": round(lon, 7),
        "source": source,
    }


def _ffprobe_video_metadata(video_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read common container/stream metadata when the ffprobe executable is installed.

    ffprobe is optional. If it is missing or the container has no metadata, sidecar
    and label lookup can still provide a location.
    """
    info: dict[str, Any] = {
        "format": video_path.suffix.lstrip(".").upper() or None,
        "duration_seconds": None,
        "width": None,
        "height": None,
        "captured_at": None,
    }
    locations: list[dict[str, Any]] = []
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return info, locations
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError):
        return info, locations

    fmt = payload.get("format") if isinstance(payload, dict) else {}
    fmt = fmt if isinstance(fmt, dict) else {}
    format_name = fmt.get("format_name")
    if format_name:
        info["format"] = str(format_name)
    try:
        duration = float(fmt.get("duration"))
        if duration >= 0:
            info["duration_seconds"] = round(duration, 3)
    except (TypeError, ValueError, OverflowError):
        pass

    tag_records: list[tuple[dict[str, Any], str]] = []
    format_tags = fmt.get("tags")
    if isinstance(format_tags, dict):
        tag_records.append((format_tags, "Embedded video container metadata (ffprobe)"))

    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    if isinstance(streams, list):
        video_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
        if video_streams:
            try:
                info["width"] = int(video_streams[0].get("width"))
                info["height"] = int(video_streams[0].get("height"))
            except (TypeError, ValueError):
                pass
        for stream in streams:
            tags = stream.get("tags") if isinstance(stream, dict) else None
            if isinstance(tags, dict):
                tag_records.append((tags, "Embedded video stream metadata (ffprobe)"))

    for tags, source in tag_records:
        for key, value in tags.items():
            norm = _normalise_key(key)
            text_value = str(value).strip() if value is not None else ""
            if norm in {"comapplequicktimelocationiso6709", "locationiso6709", "iso6709", "xyz", "gpscoordinates"}:
                loc = _location_from_iso6709(text_value, source)
                if loc:
                    locations.append(loc)
            # Some containers use `location` for an ISO 6709 string; parse it as
            # coordinates before considering it a human-readable place name.
            if norm in {"location", "locationeng", "locationund", "comapplequicktimelocationiso6709", "xyz"}:
                loc = _location_from_iso6709(text_value, source)
                if loc:
                    locations.append(loc)
            if norm in {"creationtime", "dateencoded", "datetimeoriginal", "datecreated"} and not info["captured_at"]:
                info["captured_at"] = text_value or None
        named = _location_from_record(tags, source)
        if named:
            # _location_from_record may treat a location tag containing ISO 6709
            # as a place name; avoid keeping that representation when it is coords.
            if not (named.get("name") and _location_from_iso6709(named["name"], source)):
                locations.append(named)

    # Preserve source order and remove duplicate location records.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for loc in locations:
        key = (loc.get("name"), loc.get("latitude"), loc.get("longitude"), loc.get("source"))
        if key not in seen:
            unique.append(loc)
            seen.add(key)
    return info, unique


def extract_video_metadata(
    video_path: str | Path,
    label_path: str | Path | None = None,
    search_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Collect available video metadata and an explicitly recorded location.

    Embedded metadata is read with ffprobe when it is installed. The function
    also checks matching JSON/YAML/CSV/XML/TXT sidecars and label manifests. It
    does not derive a geographic location from the video frames.
    """
    path = Path(video_path)
    media_info, candidates = _ffprobe_video_metadata(path)
    info: dict[str, Any] = {
        "filename": path.name,
        **media_info,
        "location": {
            "available": False,
            "display": "No location available",
            "name": None,
            "latitude": None,
            "longitude": None,
            "source": None,
        },
    }

    paths_to_check: list[tuple[Path, bool]] = []
    if label_path:
        explicit_path = Path(label_path)
        if explicit_path.is_file():
            paths_to_check.append((explicit_path, True))
    paths_to_check.extend((item, False) for item in _candidate_metadata_files(path, search_roots))
    seen: set[str] = set()
    for metadata_path, explicit in paths_to_check:
        try:
            key = str(metadata_path.resolve())
        except OSError:
            key = str(metadata_path)
        if key in seen:
            continue
        seen.add(key)
        result = _location_from_metadata_file(
            metadata_path,
            path,
            explicitly_selected=explicit or _is_image_sidecar(metadata_path, path),
        )
        if result:
            candidates.append(result)

    name_candidate = next((candidate for candidate in candidates if candidate.get("name")), None)
    coordinate_candidate = next((candidate for candidate in candidates if _is_valid_pair(candidate.get("latitude"), candidate.get("longitude"))), None)
    if name_candidate or coordinate_candidate:
        name = name_candidate.get("name") if name_candidate else None
        lat = coordinate_candidate.get("latitude") if coordinate_candidate else None
        lon = coordinate_candidate.get("longitude") if coordinate_candidate else None
        info["location"] = {
            "available": True,
            "display": name or (f"{lat:.6f}, {lon:.6f}" if coordinate_candidate else "No location available"),
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "source": "; ".join(dict.fromkeys(
                candidate.get("source") for candidate in candidates if candidate.get("source")
            )) or "Video metadata",
        }
    return info
