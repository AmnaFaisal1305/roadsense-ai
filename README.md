# RoadSense AI — Road Damage Detection, Tracking and Repair Prioritisation

**Course:** CT-361 Artificial Intelligence & Expert System — Complex Computing Activity (CCA)
**Group members:** _TODO: names and roll numbers_

## 1. Problem Statement

**Context.** Potholes and damaged or open manholes cause accidents, vehicle damage and traffic congestion. Road authorities usually find them through manual surveys or citizen complaints, which are slow, expensive and inconsistent, so damage is often repaired late and in no particular order.

**Objective.** Build a system that takes ordinary dash-cam video and automatically:

1. detects potholes and manholes in every frame,
2. counts each physical hazard exactly once, even though it appears in many frames,
3. ranks every hazard by repair priority, and
4. places it on a map and exports a report a maintenance crew can act on.

**Real-world significance.** A single vehicle with a camera can survey a whole route in one drive. The output is a prioritised, geo-referenced repair list rather than raw footage, so crews can be sent to the worst segments first.

**Inputs and outputs.**

| | |
|---|---|
| **Input** | Road video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) or images; route start point, bearing and vehicle speed |
| **Output** | Annotated video, unique hazard counts, per-hazard repair priority, hazard map, analytics charts, evidence snapshots, CSV / JSON / GeoJSON reports |

## 2. AI Approach (Hybrid)

The system combines three AI techniques:

| Stage | Technique | Where |
|---|---|---|
| **Detection** | Deep learning object detection (YOLO11, fine-tuned on road damage) | `src/train_640.py`, `src/train_enhanced.py` |
| **Tracking** | Kalman filter motion prediction + Hungarian algorithm matching, so one pothole seen in 40 frames is counted once | `src/tracker.py` |
| **Decision** | Rule-based expert system that turns detections into a repair priority | `priority_of()` in `app.py` |

**Expert-system rules for repair priority**

- `size = min(1, box area / 2% of the frame)` — a hazard filling 2% of the frame or more is treated as large.
- `score = 0.55 × peak confidence + 0.45 × size`
- If the hazard is a manhole, `score × 0.8` (a utility asset, not a surface failure).
- `score ≥ 0.60` → **High**, `score ≥ 0.38` → **Medium**, otherwise **Low**.

The rules are deliberately simple and visible so an inspector can see why a hazard was ranked the way it was.

## 3. Dataset

| | |
|---|---|
| **Source** | _TODO: name and link (e.g. RDD2022 / Kaggle / Roboflow) and licence_ |
| **Type** | RGB road images with YOLO-format bounding-box labels |
| **Original classes** | 0 pothole, 1 crack, 2 manhole |
| **Classes used** | 0 pothole, 1 manhole |
| **Second dataset** | _TODO: the brief asks for two datasets or one large one — add a second source or an external test set here_ |

Annotated objects per split:

| Split | Potholes | Manholes |
|---|---:|---:|
| Train | 987 | 756 |
| Validation | 135 | 95 |
| Test | 139 | 105 |

### Preprocessing (`src/prepare_dataset.py`)

- Drops malformed label lines (wrong field count, non-numeric values).
- Removes the crack class and remaps manhole from ID 2 to ID 1.
- Skips images that contained only cracks; keeps the rest, including images left with no objects, as background examples.
- Class imbalance is mild (987 potholes vs 756 manholes in training), so no re-sampling was applied.

### Augmentation (`src/train_enhanced.py`)

Chosen for the conditions that hurt road-damage detection: strong brightness and saturation jitter (shadows, overcast, wet roads), scale jitter (near vs distant damage), small rotation, shear and perspective (camera angle), mosaic (small objects), mixup and copy-paste (occlusion, more pothole instances). Vertical flips are disabled because roads are never upside down.

## 4. Models

| Role | File | Architecture | Training |
|---|---|---|---|
| **Baseline** | `best_kashan.pt` | YOLO11n | default settings (`src/train_640.py`) |
| **Improved** | `best.pt` | YOLO11s | road-specific augmentation (`src/train_enhanced.py`) |

Improvement strategies applied on top of the baseline: larger model (YOLO11n → YOLO11s), targeted augmentation, image-resolution comparison (640 vs 768), confidence-threshold tuning, and test-time augmentation (TTA).

## 5. Results

### Baseline — YOLO11n

Trained for up to 100 epochs (batch 8, patience 15, NVIDIA Tesla T4); best epoch 72, stopped at 87.

| Metric (768 × 768) | Validation | Test |
|---|---:|---:|
| Precision | 0.651 | 0.668 |
| Recall | 0.579 | 0.570 |
| mAP50 | 0.614 | 0.5913 |
| mAP50-95 | 0.283 | 0.2576 |

| Validation metric | 640 × 640 | 768 × 768 |
|---|---:|---:|
| Precision | 0.656 | 0.651 |
| Recall | 0.613 | 0.579 |
| mAP50 | 0.629 | 0.614 |
| mAP50-95 | 0.295 | 0.283 |

Raising the resolution to 768 did not help, so 640 is used.

| Class (test, 768) | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Pothole | 0.597 | 0.360 | 0.414 | 0.154 |
| Manhole | 0.738 | 0.781 | 0.768 | 0.362 |

### Baseline vs improved

_TODO: run the command below and paste the table it prints here._

```bash
python src/evaluate.py
```

It evaluates `best_kashan.pt` and `best.pt` on the test split, each with and without TTA, prints a Markdown table (including pothole recall, the baseline's weakest number) and saves confusion matrices and precision-recall curves under `runs/eval_*/`.

### Observations

- Manholes are detected much better than potholes; pothole recall is the main weakness.
- Both classes contain many small objects.
- Annotation consistency and limited scene diversity affect performance.

## 6. Privacy

Dash-cam footage contains faces and vehicle number plates. With **Blur people & vehicles** on (the default), a COCO-pretrained YOLO model finds people and vehicles and blurs them (`src/privacy.py`). Hazards are detected on the original frame, but the live preview, the saved video and the evidence snapshots are all produced from the blurred frame, so no identifiable person or plate is stored or exported.

- Uploaded videos are processed locally and written only to a temporary folder.
- Reports contain hazard data only (type, position, confidence, size) — no personal data.
- Limitation: a person or vehicle the privacy model misses is not blurred.

## 7. Ethics and Limitations

- **Transparency.** Every hazard shows its confidence, size and priority score, and the priority rules are published above. Map positions are estimated from video time, speed and bearing, not real GPS, and the app says so.
- **Accountability.** The system supports inspectors; it does not replace them. A person should confirm a hazard (using the evidence snapshot) before a repair is scheduled.
- **Fairness.** The model is only as general as its training images. Roads, lighting or regions that are under-represented in the dataset (night, rain, unpaved roads) will be detected less reliably, which could leave those areas under-served if results are used without checking.
- **Known failure cases.** Missed small or distant potholes (recall 0.36 for the baseline), shadows and patches mistaken for damage, water-filled potholes, and night-time footage.
- **External factors.** Camera angle, vehicle speed, vibration, weather and lighting all affect accuracy. Hazard density per km depends on the speed value entered by the user.
- **Network.** Detection and tracking run fully offline. A connection is needed only for map background tiles, web fonts and the one-time download of the privacy model.

## 8. Installation

```bash
git clone YOUR_REPOSITORY_URL
cd Road-Damage-Detection-main
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 9. Usage

Run every command from the project root.

| Task | Command |
|---|---|
| **Dashboard (main app)** | `streamlit run app.py` |
| Compare baseline vs improved model | `python src/evaluate.py` |
| Predict on images | `python src/predict.py --source path/to/images` |
| Predict on video | `python src/test_video.py --source tests` |
| Desktop player with tracking | `python src/player.py` |
| Build the 2-class dataset | `python src/prepare_dataset.py --classes 2` |
| Train the baseline | `python src/train_640.py` |
| Train the improved model | `python src/train_enhanced.py` |
| Export to ONNX | `python src/export_model.py` |
| Run tests | `python -m pytest tests` |

Training and evaluation need the dataset: put the original data in `road_damage_data/` (`images/` and `labels/`, each with `train`, `val`, `test`) and run `prepare_dataset.py`, which creates `road_damage_2class/`.

### Dashboard

1. Upload a road video or pick a sample from `tests/`.
2. Press **Run inspection** to watch live detection, unique counts and progress.
3. Review the results tabs: annotated video, hazard map, analytics, registry with CSV / JSON / GeoJSON export, and evidence snapshots.

## 10. Project Structure

```
app.py                   Streamlit dashboard (detection, tracking, map, analytics, export)
best.pt                  Improved model (YOLO11s)
best_kashan.pt           Baseline model (YOLO11n)
data_2class.yaml         Dataset definition
src/tracker.py           Kalman + Hungarian hazard tracker
src/privacy.py           People / vehicle blurring
src/prepare_dataset.py   Dataset cleaning and class remapping
src/train_640.py         Baseline training
src/train_enhanced.py    Improved training with road augmentations
src/evaluate.py          Baseline vs improved comparison
src/predict.py           Image prediction
src/test_video.py        Video prediction
src/player.py            Desktop player
src/export_model.py      ONNX export
tests/                   Sample video and unit tests
```

## 11. Future Work

- Raise pothole recall with more varied data (night, rain, different road types).
- Use real GPS logs instead of estimated positions.
- Combine the two models' predictions (ensembling).
- Add a dedicated number-plate detector so whole vehicles need not be blurred.
- Check the tracker's unique counts against hand counts on more videos.
