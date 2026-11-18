from pathlib import Path
import io
import json
import random
import sqlite3
import uuid
import hashlib
import os
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_v5.keras"
MODEL_METADATA_PATH = BASE_DIR / "model_metadata.json"
DB_PATH = BASE_DIR / "nagarnetra.db"
COMPLAINT_DIR = BASE_DIR / "complaints"
COMPLAINT_DIR.mkdir(exist_ok=True)
IMAGE_SIZE = (128, 128)  # legacy fallback; new trainer records its size in metadata
MODEL_IMAGE_SIZE = IMAGE_SIZE
CATEGORIES = ["potholes", "illegal_buildings", "sanitation", "garbage", "street_lights", "hanging_wires"]

app = Flask(__name__, static_folder=str(BASE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
CORS(app)

# Load model metadata and model once when the server starts.
# training; /api/health reports that the model is unavailable.
model = None
model_load_error = None
model_metadata = {}
if MODEL_METADATA_PATH.exists():
    try:
        model_metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        model_load_error = f"Could not read model metadata: {exc}"

MODEL_IMAGE_SIZE = tuple(model_metadata.get("image_size", IMAGE_SIZE))

if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        metadata_categories = model_metadata.get("categories")
        if metadata_categories and metadata_categories != CATEGORIES:
            raise ValueError("Model metadata class order does not match app.py CATEGORIES.")
    except Exception as exc:
        model_load_error = str(exc)
else:
    model_load_error = f"Model file not found: {MODEL_PATH.name}"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ref_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                phone TEXT,
                category TEXT NOT NULL,
                department TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                landmark TEXT,
                confidence REAL,
                dimensions_json TEXT,
                audit_json TEXT,
                status TEXT NOT NULL DEFAULT 'RECORDED',
                created_at TEXT NOT NULL,
                applicant_name TEXT,
                applicant_phone TEXT,
                public_name TEXT,
                public_phone TEXT,
                redacted_application_path TEXT,
                application_sha256 TEXT,
                evidence_sha256 TEXT,
                destination_type TEXT,
                destination_email TEXT,
                dispatch_status TEXT NOT NULL DEFAULT 'NOT_SENT',
                dispatch_message TEXT
            )
        """)


init_db()


def migrate_db():
    additions = {
        "applicant_name": "TEXT", "applicant_phone": "TEXT",
        "public_name": "TEXT", "public_phone": "TEXT",
        "redacted_application_path": "TEXT", "application_sha256": "TEXT",
        "evidence_sha256": "TEXT", "destination_type": "TEXT",
        "destination_email": "TEXT",
        "dispatch_status": "TEXT NOT NULL DEFAULT 'NOT_SENT'",
        "dispatch_message": "TEXT"
    }
    with get_db() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(complaints)")}
        for col, declaration in additions.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE complaints ADD COLUMN {col} {declaration}")


migrate_db()


def safe_text(value, limit=2000):
    return str(value or "").strip()[:limit]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_application_pdf(path, record):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=48, leftMargin=48,
                            topMargin=48, bottomMargin=48)
    story = [
        Paragraph("<b>NAGARNETRA</b>", styles["Title"]),
        Paragraph("CIVIC COMPLAINT APPLICATION", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph(f"<b>Reference:</b> {record['ref_id']}", styles["BodyText"]),
        Paragraph(f"<b>Date:</b> {record['created_at']}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("<b>To,</b><br/>The Officer-in-Charge / Competent Officer<br/>" + record["department"] + "<br/>" + (record.get("area_name") or "Jurisdiction to be verified"), styles["BodyText"]),
        Spacer(1, 10),
        Paragraph(f"<b>Subject:</b> Formal application requesting inspection and necessary action regarding {record['category'].replace('_', ' ')} at the location stated below (Ref: {record['ref_id']})", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Respected Sir/Madam,", styles["BodyText"]),
        Spacer(1, 6),
        Paragraph("I respectfully submit this application to bring to your notice a civic issue observed at the location mentioned below. I request your office to kindly verify the reported condition, arrange an inspection through the concerned official, and take appropriate action in accordance with applicable rules and procedures. If this matter falls outside your jurisdiction, I request that it be forwarded to the competent office or that the correct authority be communicated to me.", styles["BodyText"]),
        Spacer(1, 12)
    ]
    rows = [
        ["Category", record["category"].replace("_", " ").title()],
        ["Location / area", record.get("area_name") or "GPS coordinates (jurisdiction not confirmed)"],
        ["GPS coordinates", f"{record['latitude']:.6f}, {record['longitude']:.6f}"],
        ["Landmark", record.get("landmark") or "Not provided"],
        ["Applicant name", "XYZ"],
        ["Applicant phone", "XYZ"],
        ["AI confidence", str(record.get("confidence") if record.get("confidence") is not None else "Not available")],
        ["Evidence SHA-256", record.get("evidence_sha256") or "Not supplied"]
    ]
    table = Table(rows, colWidths=[145, 335])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf0f7")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
    ]))
    story.extend([
        table, Spacer(1, 16),
        Paragraph("I shall be grateful for an acknowledgement and the official complaint / diary reference number, along with an update on the action taken. I am enclosing the available photographic evidence through the NagarNetra record.", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Thank you.", styles["BodyText"]),
        Spacer(1, 18), Paragraph("Yours faithfully,<br/>XYZ<br/>Applicant (identity masked in this copy)", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("This application is generated by a prototype and is not proof of receipt or acceptance by the authority.", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Integrity note: SHA-256 supports file-integrity checking only; it does not prove authorship or official receipt.", styles["Italic"])
    ])
    doc.build(story)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum upload size is 8 MB."}), 413


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "model_architecture": model_metadata.get("architecture", "Legacy model (metadata unavailable)"),
        "real_photo_model": model_metadata.get("synthetic_data_used") is False,
        "confidence_threshold": float(model_metadata.get("confidence_threshold", 0.60))
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({
            "error": "The AI model is not available. Train a model and restart Flask.",
            "model_error": model_load_error
        }), 503

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Please upload an image using the 'image' field."}), 400

    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB").resize(MODEL_IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32)
        # New MobileNetV2 model embeds preprocessing. Legacy CNN expects [0,1].
        if not model_metadata.get("input_format", "").startswith("RGB float32 pixels in range 0..255"):
            array = array / 255.0

        prediction = model.predict(np.expand_dims(array, axis=0), verbose=0)
        scores = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != len(CATEGORIES):
            return jsonify({"error": "Model output shape does not match the six configured categories."}), 500

        probabilities = scores[0].astype(float)
        if not np.all(np.isfinite(probabilities)):
            return jsonify({"error": "Model returned invalid probability values."}), 500
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        threshold = float(model_metadata.get("confidence_threshold", 0.60))
        selected_category = request.form.get("selected_category", "").strip()
        if selected_category not in CATEGORIES:
            selected_category = None

        uncertain = confidence < threshold
        chosen_category = selected_category if uncertain and selected_category else CATEGORIES[best_index]
        return jsonify({
            "category": chosen_category,
            "predicted_category": CATEGORIES[best_index],
            "confidence": round(confidence * 100, 2),
            "needs_review": uncertain,
            "confidence_threshold": round(threshold * 100, 1),
            "review_message": (
                f"AI confidence is below {threshold * 100:.0f}%. Please verify the category manually. "
                f"The model's top suggestion was {CATEGORIES[best_index].replace('_', ' ')}."
            ) if uncertain else None,
            "model_architecture": model_metadata.get("architecture", "Legacy model")
        })
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "The uploaded file is not a valid image."}), 400
    except Exception as exc:
        app.logger.exception("Image analysis failed")
        return jsonify({"error": f"Image analysis failed: {str(exc)}"}), 500




@app.route("/api/nearby-authorities", methods=["GET"])
def nearby_authorities():
    """Reverse-geocode GPS and search for publicly listed authority contacts.
    Results are leads only; the user must verify the source before using an address.
    """
    try:
        lat = float(request.args.get("lat", ""))
        lon = float(request.args.get("lng", ""))
    except (TypeError, ValueError):
        return jsonify({"error": "Valid GPS latitude and longitude are required."}), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "Coordinates outside valid ranges."}), 400

    import requests
    from ddgs import DDGS
    headers = {"User-Agent": "NagarNetraCivicPrototype/1.0 (local civic complaint helper)"}
    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1},
            headers=headers, timeout=12
        )
        geo.raise_for_status()
        geo_data = geo.json()
        address = geo_data.get("address", {})
        locality = (address.get("city") or address.get("town") or address.get("municipality")
                    or address.get("county") or address.get("village") or "")
        district = address.get("state_district") or address.get("county") or ""
        state = address.get("state") or ""
        country = address.get("country") or ""
        place = ", ".join(x for x in [locality, district, state, country] if x)
        if not place:
            place = geo_data.get("display_name", "GPS location")
    except Exception as exc:
        app.logger.warning("Reverse geocoding failed: %s", exc)
        return jsonify({"error": "Could not identify the GPS area. Check internet access and try again."}), 502

    category = safe_text(request.args.get("category"), 50)
    destination = safe_text(request.args.get("destination"), 30).lower()
    category_department = {
        "potholes": "municipal corporation roads department",
        "illegal_buildings": "municipal corporation town planning department",
        "sanitation": "municipal corporation sanitation drainage department",
        "garbage": "municipal corporation solid waste department",
        "street_lights": "municipal corporation street light department",
        "hanging_wires": "electricity distribution company electrical safety office"
    }.get(category, "municipal corporation public grievance office")
    target = {
        "municipal": category_department,
        "police": "district police superintendent office police station",
        "rti": "public information officer RTI",
        "electricity": "electricity distribution company electrical safety office",
        "email_only": "official public grievance contact"
    }.get(destination, category_department)

    results = []
    queries = [
        f"{target} {place} official email contact",
        f"{target} {place} site:gov.in email",
        f"{target} {place} site:nic.in email"
    ]
    seen = set()
    try:
        with DDGS() as ddgs:
            for query in queries:
                for item in ddgs.text(query, region="in-en", max_results=6):
                    title = safe_text(item.get("title"), 240)
                    url = safe_text(item.get("href"), 600)
                    snippet = safe_text(item.get("body"), 600)
                    text_blob = " ".join([title, snippet])
                    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text_blob)
                    for email in emails:
                        key = email.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append({
                            "email": email,
                            "title": title,
                            "source_url": url,
                            "snippet": snippet,
                            "verification_required": True
                        })
    except Exception as exc:
        app.logger.warning("Official contact search failed: %s", exc)

    return jsonify({
        "location": place,
        "latitude": lat,
        "longitude": lon,
        "results": results[:12],
        "notice": "Search results are unverified leads, not endorsements. Open the source and confirm the address belongs to the correct jurisdiction before selecting it."
    })


@app.route("/api/dispatch", methods=["POST"])
def dispatch():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON request body is required."}), 400
    category = data.get("category")
    if category not in CATEGORIES:
        return jsonify({"error": "Invalid or missing category."}), 400
    try:
        latitude, longitude = float(data.get("lat")), float(data.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "Valid latitude and longitude are required."}), 400
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return jsonify({"error": "Coordinates outside valid ranges."}), 400

    departments = {
        "potholes": "Municipal Roads Department (verify local jurisdiction)",
        "illegal_buildings": "Municipal Building / Town Planning Enforcement (suspected issue; human verification required)",
        "sanitation": "Municipal Sanitation / Drainage Department",
        "garbage": "Municipal Solid Waste / Sanitation Department",
        "street_lights": "Municipal Electrical / Street Lighting Department",
        "hanging_wires": "Relevant Electricity Distribution / Electrical Safety Office"
    }
    destination_type = safe_text(data.get("destination_type"), 30).lower() or "municipal"
    if destination_type not in {"municipal", "police", "rti", "electricity", "email_only"}:
        return jsonify({"error": "Unsupported destination type."}), 400

    department = departments[category]
    if destination_type == "police":
        department = "Police Station / SP Office (verify jurisdiction)"
    elif destination_type == "rti":
        department = "Public Information Officer of the relevant public authority"
    elif destination_type == "electricity":
        department = "Relevant Electricity Distribution Office"

    ref_id = f"NN-{uuid.uuid4().hex[:10].upper()}"
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    applicant_name = safe_text(data.get("name"), 100) or "Anonymous"
    applicant_phone = safe_text(data.get("phone"), 30)
    landmark = safe_text(data.get("landmark"), 200)
    try:
        confidence = round(float(data.get("confidence")), 2) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    evidence_sha256 = safe_text(data.get("evidence_sha256"), 64) or None
    dimensions = data.get("dimensions") if isinstance(data.get("dimensions"), dict) else {}

    record = {
        "ref_id": ref_id, "created_at": created_at, "department": department,
        "category": category, "latitude": latitude, "longitude": longitude,
        "landmark": landmark, "confidence": confidence,
        "area_name": safe_text(data.get("area_name"), 300),
        "evidence_sha256": evidence_sha256
    }
    pdf_path = COMPLAINT_DIR / f"{ref_id}_application.pdf"
    build_application_pdf(pdf_path, record)
    application_hash = sha256_file(pdf_path)

    with get_db() as conn:
        conn.execute("""
            INSERT INTO complaints (
                ref_id, name, phone, category, department, latitude, longitude,
                landmark, confidence, dimensions_json, audit_json, status, created_at,
                applicant_name, applicant_phone, public_name, public_phone,
                redacted_application_path, application_sha256, evidence_sha256,
                destination_type, destination_email, dispatch_status, dispatch_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ref_id, "XYZ", "XYZ", category, department, latitude, longitude,
            landmark, confidence, json.dumps(dimensions), None, "RECORDED", created_at,
            applicant_name, applicant_phone, "XYZ", "XYZ", str(pdf_path), application_hash,
            evidence_sha256, destination_type,
            safe_text(data.get("destination_email"), 254) or None,
            "NOT_SENT", "PDF created and saved locally; no external authority has been contacted."
        ))

    return jsonify({
        "status": "success", "ref_id": ref_id, "dispatched_to": department,
        "application_url": f"/api/complaints/{ref_id}/application",
        "verification_url": f"/api/complaints/{ref_id}/verify",
        "application_sha256": application_hash, "dispatch_status": "NOT_SENT",
        "message": "Masked PDF generated and complaint saved locally. No email or government submission has been made."
    }), 201


@app.route("/api/complaints/<ref_id>/application", methods=["GET"])
def download_application(ref_id):
    with get_db() as conn:
        row = conn.execute("SELECT redacted_application_path FROM complaints WHERE ref_id = ?", (ref_id,)).fetchone()
    if row is None or not row["redacted_application_path"]:
        return jsonify({"error": "Application not found."}), 404
    path = Path(row["redacted_application_path"]).resolve()
    if not path.exists() or path.parent != COMPLAINT_DIR.resolve():
        return jsonify({"error": "Application file missing or invalid."}), 404
    return send_file(path, mimetype="application/pdf", as_attachment=True,
                     download_name=f"{ref_id}_application.pdf")


@app.route("/api/complaints/<ref_id>/verify", methods=["GET"])
def verify_application(ref_id):
    with get_db() as conn:
        row = conn.execute("SELECT redacted_application_path, application_sha256 FROM complaints WHERE ref_id = ?", (ref_id,)).fetchone()
    if row is None or not row["redacted_application_path"]:
        return jsonify({"error": "Application not found."}), 404
    path = Path(row["redacted_application_path"]).resolve()
    if not path.exists():
        return jsonify({"verified": False, "reason": "PDF file is missing."}), 404
    actual_hash = sha256_file(path)
    return jsonify({
        "ref_id": ref_id, "verified": actual_hash == row["application_sha256"],
        "algorithm": "SHA-256", "stored_sha256": row["application_sha256"],
        "current_sha256": actual_hash,
        "meaning": "Integrity check only; not proof of authorship or official receipt."
    })


@app.route("/api/complaints/<ref_id>/email", methods=["POST"])
def email_complaint(ref_id):
    # Explicit, operator-triggered email only; no recipient is assumed.
    import smtplib
    from email.message import EmailMessage
    data = request.get_json(silent=True) or {}
    recipient = safe_text(data.get("recipient"), 254)
    if not recipient or "@" not in recipient:
        return jsonify({"error": "A verified recipient email is required."}), 400

    host = os.environ.get("NAGARNETRA_SMTP_HOST")
    port = int(os.environ.get("NAGARNETRA_SMTP_PORT", "587"))
    user = os.environ.get("NAGARNETRA_SMTP_USER")
    password = os.environ.get("NAGARNETRA_SMTP_PASSWORD")
    sender = os.environ.get("NAGARNETRA_FROM_EMAIL") or user
    if not all([host, user, password, sender]):
        return jsonify({"error": "SMTP is not configured on the server."}), 503

    with get_db() as conn:
        row = conn.execute("SELECT redacted_application_path, category, department FROM complaints WHERE ref_id = ?", (ref_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Complaint reference not found."}), 404
    pdf_path = Path(row["redacted_application_path"])
    if not pdf_path.exists():
        return jsonify({"error": "Application PDF is missing."}), 404

    message = EmailMessage()
    message["Subject"] = f"NagarNetra complaint {ref_id}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        f"Please review the attached masked NagarNetra complaint.\\nReference: {ref_id}\\n"
        f"Category: {row['category']}\\nRouting label: {row['department']}\\n"
        "Please confirm receipt through your official process."
    )
    message.add_attachment(pdf_path.read_bytes(), maintype="application", subtype="pdf",
                           filename=f"{ref_id}_application.pdf")
    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(message)
    except Exception as exc:
        app.logger.exception("Email dispatch failed")
        return jsonify({"error": str(exc), "dispatch_status": "FAILED"}), 502

    with get_db() as conn:
        conn.execute("UPDATE complaints SET dispatch_status = ?, dispatch_message = ? WHERE ref_id = ?",
                     ("EMAIL_SENT", f"SMTP accepted the message for {recipient}", ref_id))
    return jsonify({
        "status": "success", "dispatch_status": "EMAIL_SENT",
        "message": "SMTP accepted the email; this is not proof of government registration."
    })


@app.route("/api/complaints/<ref_id>", methods=["GET"])
def get_complaint(ref_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT ref_id, category, department, latitude, longitude, landmark, status, created_at, application_sha256, destination_type, dispatch_status "
            "FROM complaints WHERE ref_id = ?",
            (ref_id,)
        ).fetchone()
    if row is None:
        return jsonify({"error": "Complaint reference not found."}), 404
    return jsonify(dict(row))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
