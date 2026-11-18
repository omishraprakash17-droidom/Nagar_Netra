"""
NagarNetra free image collector — DuckDuckGo image search via ddgs.

No paid API key is required. This downloads candidate images into:
dataset/real/<class>/

Install/run:
    python -m pip install -r requirements.txt
    python civic_dataset_downloader.py

Notes:
- DuckDuckGo image results are third-party search results, not verified labels.
- Search results may be irrelevant, non-Indian, duplicated, or unavailable.
- Search results do not grant copyright permission. Check each source and licence.
- Respect provider rate limits; do not run repeatedly in rapid succession.
"""

from pathlib import Path
from io import BytesIO
from urllib.parse import urlparse
import csv
import hashlib
import re
import time
import requests
from PIL import Image, UnidentifiedImageError
from ddgs import DDGS

PROJECT_DIR = Path(__file__).resolve().parent
OUT = PROJECT_DIR / "dataset" / "real"
ZIP_PATH = PROJECT_DIR / "Civic_Dataset.zip"
TARGET_PER_CLASS = 10
MAX_RESULTS_PER_QUERY = 25
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MIN_WIDTH, MIN_HEIGHT = 240, 160
USER_AGENT = "Mozilla/5.0 (compatible; NagarNetraEducationalDataset/1.0)"

CLASSES = {
    "potholes": [
        "pothole damaged asphalt road India",
        "large road pothole Indian street",
        "broken road surface potholes India",
    ],
    "illegal_buildings": [
        "building encroachment India urban street",
        "unauthorized construction India building",
        "roadside building encroachment India",
    ],
    "sanitation": [
        "open sewage drain Indian street",
        "overflowing open drainage India urban",
        "poor sanitation public street India",
    ],
    "garbage": [
        "roadside garbage dump Indian street",
        "municipal waste pile street India",
        "litter garbage overflowing bin India street",
    ],
    "street_lights": [
        "broken street light pole India road",
        "damaged street lamp Indian street",
        "non functioning streetlight urban India",
    ],
    "hanging_wires": [
        "low hanging electrical wires Indian street",
        "tangled overhead electric cables India road",
        "exposed hanging wires urban India",
    ],
}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
seen_hashes = set()
source_rows = []


def safe_filename(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (value[:80] or "image")


def load_existing_hashes():
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                seen_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                pass


def search_images(query):
    # ddgs is a free community Python client; no API key is needed.
    with DDGS() as ddgs:
        return list(ddgs.images(
            query,
            region="in-en",
            safesearch="moderate",
            max_results=MAX_RESULTS_PER_QUERY,
        ))


def download_candidate(result, query, class_dir):
    image_url = result.get("image")
    if not image_url:
        return False
    try:
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            return False

        with session.get(image_url, stream=True, timeout=35) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not content_type.startswith("image/"):
                return False
            chunks, total = [], 0
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    return False
                chunks.append(chunk)
            content = b"".join(chunks)

        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_hashes:
            return False

        image = Image.open(BytesIO(content))
        image.verify()
        image = Image.open(BytesIO(content)).convert("RGB")
        if image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
            return False

        seen_hashes.add(digest)
        stem = safe_filename(result.get("title") or "search_result")
        index = len(list(class_dir.glob("*.jpg"))) + 1
        destination = class_dir / f"{index:03d}_{stem}.jpg"
        while destination.exists():
            index += 1
            destination = class_dir / f"{index:03d}_{stem}.jpg"
        image.save(destination, "JPEG", quality=92, optimize=True)

        source_rows.append({
            "filename": destination.relative_to(OUT).as_posix(),
            "class": class_dir.name,
            "title": result.get("title", ""),
            "source": result.get("source", ""),
            "source_page_url": result.get("url", ""),
            "image_url": image_url,
            "search_query": query,
            "width": image.width,
            "height": image.height,
            "sha256": digest,
            "license_status": "UNVERIFIED - check source and permission before training",
        })
        print(f"    + {destination.name} ({image.width}x{image.height})")
        return True
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError) as exc:
        print(f"    skipped {result.get('title', 'result')}: {exc}")
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load_existing_hashes()

    csv_path = OUT / "sources.csv"
    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
    known = {row.get("filename") for row in existing_rows}

    for class_name, queries in CLASSES.items():
        class_dir = OUT / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        count = len(list(class_dir.glob("*.jpg")))
        print(f"\n[{class_name}] {count}/{TARGET_PER_CLASS}")

        for query in queries:
            if count >= TARGET_PER_CLASS:
                break
            print(f"  Searching: {query}")
            try:
                results = search_images(query)
            except Exception as exc:
                print(f"  Search failed (possibly rate-limited): {exc}")
                time.sleep(3)
                continue

            for result in results:
                if count >= TARGET_PER_CLASS:
                    break
                if download_candidate(result, query, class_dir):
                    count += 1
                time.sleep(0.3)
            time.sleep(1.5)

        print(f"  Downloaded: {count}/{TARGET_PER_CLASS}")
        if count < TARGET_PER_CLASS:
            print("  Search returned fewer usable images; try again later or add reviewed images manually.")

    fields = [
        "filename", "class", "title", "source", "source_page_url",
        "image_url", "search_query", "width", "height", "sha256", "license_status",
    ]
    combined = existing_rows + [r for r in source_rows if r["filename"] not in known]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)

    (OUT / "REVIEW_BEFORE_TRAINING.txt").write_text(
        "Review every downloaded image before training. Confirm the image depicts "
        "the intended category and preferably an Indian civic setting. Search "
        "engines do not grant copyright permission. Check the source page and "
        "licence/permission for each file; sources.csv marks all files UNVERIFIED. "
        "Delete incorrect, duplicate, watermarked, misleading or unlicensed images. "
        "A photograph cannot establish that a building is illegal.\n",
        encoding="utf-8",
    )

    import zipfile
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT_DIR))
    print(f"\nDataset: {OUT}")
    print(f"ZIP: {ZIP_PATH}")
    print("Check image relevance and permissions before model training.")


if __name__ == "__main__":
    main()
