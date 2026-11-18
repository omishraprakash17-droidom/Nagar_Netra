# NagarNetra V3 — real-photo transfer-learning pipeline

## What changed
- MobileNetV2 transfer learning replaces the small CNN trainer.
- Input preprocessing is embedded in the saved model to keep training and inference consistent.
- Stratified train/validation/test splits, class weights, early stopping and fine-tuning are included.
- Model metadata records class order, preprocessing and confidence threshold.
- Low-confidence predictions require manual review.
- Misleading Depth, Length and Width UI cards have been removed.
- V2 PDF generation, masking, SQLite, SHA-256 verification and SMTP handoff are retained.
- Synthetic sample output is stored separately and is not used by the real-photo trainer.

## Real dataset structure
Create:

```text
dataset/real/
  pothole/
  construction/
  garbage/
  wires/
  streetlight/
  good_place/
```

Put real, correctly labelled JPG/JPEG/PNG/BMP/WEBP photographs in each folder. At least 10 per class is a technical minimum only; collect hundreds or thousands of varied images per class for meaningful evaluation. Avoid duplicate or near-duplicate photos across splits. For a stronger test, split by location or capture session.

Do not run `generate_data.py` for model training. It creates synthetic smoke-test images only.

## Install and train (PowerShell)
```powershell
python -m pip install -r requirements.txt
python train_model.py
```

The first run downloads ImageNet MobileNetV2 weights and therefore needs internet access. If memory is constrained:
```powershell
python train_model.py --head-epochs 8 --fine-tune-epochs 4
```

The trainer creates `model_v5.keras` and `model_metadata.json`. Copy both files together if moving the model.

## Start application
Stop the old Flask process with Ctrl+C, then:
```powershell
python app.py
```
Open http://127.0.0.1:5000/ and check http://127.0.0.1:5000/api/health.

## Confidence handling
Default threshold is 60%. Lower-confidence predictions trigger a manual-review warning and retain the user's category selection. This is a workflow threshold, not a calibrated probability guarantee; tune it on an independent validation set and inspect per-class precision/recall.

## Limitations
- No real dataset is included and no new trained weights are bundled. The new model can only be trained after real labelled photos are added.
- A legacy model without `model_metadata.json` uses the former 0..1 preprocessing path. It may run, but its real-world predictions remain unvalidated.
- Image classification cannot measure pothole depth, length or width. Those fields have been removed. Measurements need calibrated depth sensing or photogrammetry.
- Verify predictions manually before routing or filing complaints.
- Email and government portal delivery remain separately configured; route selection alone is not submission.


## Download candidate real photographs from Wikimedia Commons

The project includes `civic_dataset_downloader.py`. It downloads up to 10 candidate images per class (where Commons search and licence metadata allow) directly into the same `dataset/real/` folders used by `train_model.py`.

Run from the project folder in PowerShell:

```powershell
python -m pip install requests pillow
python civic_dataset_downloader.py
```

It creates `Civic_Dataset.zip` in the project folder. The images are also stored in `dataset/real/<class>/`, ready for manual review and possible training use.

**Important:** Search matches are not guaranteed to be Indian or correctly labelled. Open `dataset/real/sources.csv`, check the original Commons page and licence, and manually inspect each image. Remove incorrect or duplicate images before training. The downloader only accepts a limited list of explicit reusable licence labels; it may find fewer than 10 images for a class. Ten images per class is only the script's technical minimum, not enough for a reliable production classifier.

The trainer and app now use the same six class labels:
`potholes`, `illegal_buildings`, `sanitation`, `garbage`, `street_lights`, `hanging_wires`.

The illegal-buildings class describes suspected visible encroachment/unauthorized construction only. The model must not make legal determinations.


## Search-engine image downloader (Brave Image Search)

The downloader now uses targeted image-search queries instead of Wikimedia Commons file search. It writes candidate images directly into the six `dataset/real/` class folders and creates `Civic_Dataset.zip` plus `sources.csv`.

1. Create a Brave Search API key from the Brave Search API dashboard.
2. In PowerShell, from this project directory:

```powershell
python -m pip install requests pillow
$env:BRAVE_API_KEY = "YOUR_BRAVE_SEARCH_API_KEY"
python civic_dataset_downloader.py
```

The key is read from the environment and is not stored in the source code. The downloader targets India and uses category-specific search phrases. Search results can still be irrelevant or outside India.

**Copyright and quality:** the search API does not grant image-use rights. Each image is marked `UNVERIFIED` in `dataset/real/sources.csv`. Inspect each image, open its source page, verify the licence/permission, and remove irrelevant or unlicensed items before training. Do not treat search relevance as ground truth. The suspected `illegal_buildings` class is not a legal determination.


## Free image search (no API key)

The downloader now uses the `ddgs` Python package to retrieve DuckDuckGo image-search results. It does not require a paid API key or Brave account.

```powershell
python -m pip install -r requirements.txt
python civic_dataset_downloader.py
```

Images are saved directly to `dataset/real/<class>/`, and a `Civic_Dataset.zip` plus `sources.csv` are created. This is a free, unofficial/community-client approach and can be rate-limited or change if the search provider changes. Do not run it continuously.

Search relevance is not guaranteed. Inspect every image and verify its source and reuse rights; search engines do not grant copyright permission. All downloaded images are marked UNVERIFIED in sources.csv until checked.


## Complaint destination and Gmail draft

On the submission screen, select Municipal Authority, Police Station / SP Office,
RTI Public Information Officer, or Electricity Distribution Office. Enter the
recipient's email only after verifying it from an official source. After the
complaint record and masked PDF are created, click **Open Gmail with this complaint**.
The app prepares a Gmail compose draft with the destination, issue category,
location, reference number, and SHA-256 hash. Review the recipient and message,
download the generated PDF, attach it manually in Gmail, and send it yourself.

The app does not discover or invent official email addresses, send email without
your action, or claim that a Gmail message is an officially registered complaint.
For RTI, the prepared draft requests information/action from the relevant Public
Information Officer; verify the correct public authority and RTI process.


## GPS-based nearby authority contact lookup

On the complaint submission step, choose the destination type and click
**Find nearby official email using GPS**. Allow browser location permission.
NagarNetra reverse-geocodes the coordinates to an area and searches public
web results for relevant authority contact addresses. Each result includes
its source page. Open the source and verify the email and jurisdiction before
selecting it. Search results are unverified leads and can be incomplete or stale.
If no address is found, use the authority's official website to locate the
correct address and enter it manually.

The Gmail draft is written as a formal application/complaint with a subject,
officer/department, location, reference number, request for action, and closing.
For RTI, the draft uses an information-request format. Attach the downloaded
PDF manually in Gmail. The app does not send the email automatically and does
not guarantee that a draft email constitutes a valid RTI filing or registered
government complaint. GPS is used only after the user clicks the lookup button.
