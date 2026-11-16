import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Local cache directories to avoid Windows permission issues
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import altair as alt
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import torch
from src.privacy import PRIVACY_MODEL, anonymize
from src.tracker import RoadHazardKalmanTracker
from ultralytics import YOLO

# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="RoadSense AI · Hazard Inspection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# Theme system
# ==================================================
THEMES = {
    "dark": dict(
        bg="#131518", bg2="#0F1114", surface="#1A1D21", surface2="#22262B", border="#2E3239",
        text="#E8E6E0", muted="#9A9EA5", faint="#6C7078", accent="#F2B705", accent_ink="#111111",
        road="#0C0D10", pothole="#F0554A", manhole="#4BB3DA", crack="#D9B93B", ok="#5CC183", violet="#E8E6E0",
        grid="#2E3239", shadow="none", map="dark",
    ),
    "light": dict(
        bg="#E9E7E1", bg2="#E0DED7", surface="#F4F2EC", surface2="#ECE9E2", border="#C9C6BC",
        text="#1A1C1F", muted="#5A5E64", faint="#8C9096", accent="#E7A900", accent_ink="#111111",
        road="#2A2D32", pothole="#BE3027", manhole="#17688A", crack="#8A6D00", ok="#2F7D4F", violet="#1A1C1F",
        grid="#C9C6BC", shadow="none", map="light",
    ),
}

# BGR colours used when drawing on video frames (theme-independent, high contrast)
DRAW_BGR = {
    "pothole": (74, 85, 240),
    "manhole": (218, 179, 75),
    "crack": (59, 185, 217),
}
CLASS_ICON = {"pothole": "🔴", "manhole": "🔵", "crack": "🟡"}

# Header illustration: what the system sees through the dash-cam (a car ahead, a pothole and a manhole, each boxed)
HERO_SCENE_SVG = (
    '<svg viewBox="0 0 480 270" role="img" aria-label="Dash-cam view of a road with a car ahead, a detected pothole and a detected manhole" font-family="Barlow, sans-serif">'
    '<rect width="480" height="270" fill="#39414b"/><rect y="118" width="480" height="152" fill="#1b1f25"/>'
    '<polygon points="190,118 290,118 470,270 10,270" fill="#2f343b"/>'
    '<path d="M190 118L10 270M290 118L470 270" stroke="#F4F2EC" stroke-width="2" opacity=".7"/>'
    '<path d="M240 118V270" stroke="#E7A900" stroke-width="3" stroke-dasharray="16 14"/>'
    # car ahead, seen from behind
    '<g transform="translate(254,92) scale(1)">'
    '<path d="M14 30L24 4H66L76 30Z" fill="#b9b6ac"/><path d="M21 28L28 9H62L69 28Z" fill="#2b3340"/>'
    '<rect x="0" y="28" width="90" height="46" rx="6" fill="#d9d6cc"/>'
    '<rect x="5" y="38" width="18" height="9" fill="#F0554A"/><rect x="67" y="38" width="18" height="9" fill="#F0554A"/>'
    '<rect x="31" y="50" width="28" height="10" fill="#F4F2EC" stroke="#8c9096"/>'
    '<rect x="0" y="64" width="90" height="10" rx="3" fill="#8c9096"/>'
    '<rect x="6" y="72" width="17" height="14" rx="2" fill="#0d0f12"/><rect x="67" y="72" width="17" height="14" rx="2" fill="#0d0f12"/></g>'
    # pothole, boxed
    '<path d="M126 228c6-10 22-12 36-9s26 4 24 12-20 9-34 8-30-3-26-11z" fill="#0d0f12"/>'
    '<rect x="116" y="208" width="80" height="40" fill="none" stroke="#F0554A" stroke-width="2"/>'
    '<rect x="116" y="194" width="104" height="14" fill="#F0554A"/>'
    '<text x="121" y="205" font-size="11" font-weight="600" fill="#fff">pothole 14  0.83</text>'
    '<ellipse cx="205" cy="160" rx="10" ry="3" fill="#0d0f12"/>'
    # manhole, boxed
    '<ellipse cx="356" cy="232" rx="24" ry="8" fill="#555c66" stroke="#0d0f12" stroke-width="2"/>'
    '<path d="M338 232h36M356 225v14M345 227l22 10M367 227l-22 10" stroke="#0d0f12" stroke-width="1.2"/>'
    '<rect x="326" y="218" width="60" height="28" fill="none" stroke="#4BB3DA" stroke-width="2"/>'
    '<rect x="326" y="205" width="96" height="13" fill="#4BB3DA"/>'
    '<text x="331" y="215" font-size="11" font-weight="600" fill="#06202e">manhole 3  0.91</text>'
    '<text x="12" y="22" font-size="12" font-weight="600" fill="#F4F2EC">DASH-CAM VIEW</text>'
    '<text x="468" y="22" font-size="12" fill="#F4F2EC" text-anchor="end">2 hazards in frame</text></svg>'
)

# Baseline YOLO11n results reported in the README (test split unless noted)
BASELINE_ACCURACY = [("Precision", "Pothole", 0.597), ("Recall", "Pothole", 0.360), ("mAP50", "Pothole", 0.414),
                     ("Precision", "Manhole", 0.738), ("Recall", "Manhole", 0.781), ("mAP50", "Manhole", 0.768)]
DATASET_OBJECTS = [("Train", "Pothole", 987), ("Train", "Manhole", 756), ("Validation", "Pothole", 135),
                   ("Validation", "Manhole", 95), ("Test", "Pothole", 139), ("Test", "Manhole", 105)]
RESOLUTION_VAL = [("Precision", "640 px", 0.656), ("Recall", "640 px", 0.613), ("mAP50", "640 px", 0.629),
                  ("Precision", "768 px", 0.651), ("Recall", "768 px", 0.579), ("mAP50", "768 px", 0.614)]
PRIORITY_ORDER = ["High", "Medium", "Low"]

# Must be the very first rule in the injected <style>, or browsers ignore it
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');"

BASE_CSS = """
html, body, .stApp { color: var(--text); font-family: Barlow, system-ui, sans-serif; font-size: 15px; }
.stApp { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 1.6rem 2.1rem 3rem; max-width: 1520px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--text); font-family: 'Barlow Condensed', Barlow, sans-serif; }
.stApp p, .stApp li, .stApp label, [data-testid="stWidgetLabel"] p, .stMarkdown { color: var(--text); font-family: Barlow, system-ui, sans-serif; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 13px; }
hr { border-color: var(--border) !important; }
.num, .mono { font-variant-numeric: tabular-nums; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--text); font-size: 14px; font-weight: 400; }
.brand { font: 700 22px 'Barlow Condensed', sans-serif; line-height: 1; color: var(--text); padding: 2px 0 8px; }
.brand small { display: block; font: 400 13px Barlow, sans-serif; color: var(--muted); margin-top: 4px; }
.side-sec { font: 600 13px Barlow, sans-serif; color: var(--muted); margin: 22px 0 6px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.row { display: flex; justify-content: space-between; gap: 10px; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 14px; color: var(--text); }
.row span:last-child { color: var(--muted); text-align: right; }

/* ---------- Inputs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"],
.stNumberInput input, .stTextInput input {
  background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 2px !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--text) !important; }
[data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: var(--surface) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover { background: var(--bg2) !important; }
[data-baseweb="tag"] { background: var(--bg2) !important; border: 1px solid var(--border); border-radius: 2px !important; }
[data-baseweb="tag"] span { color: var(--text) !important; }
.stNumberInput button { background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; font-variant-numeric: tabular-nums; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 1px solid var(--text) !important; border-radius: 0 !important;
  width: 9px !important; height: 14px !important; box-shadow: none !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1px dashed var(--border) !important; border-radius: 2px; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--bg2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--text) !important; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 2px; border: 1px solid var(--border); background: var(--surface); color: var(--text);
  font-weight: 600; padding: .45rem 1rem; box-shadow: none; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--text); color: var(--text); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--accent) !important; color: var(--accent_ink) !important; border: none !important; box-shadow: none !important; }
.stButton > button[kind="primary"] p, .stDownloadButton > button[kind="primary"] p {
  color: var(--accent_ink) !important; font: 700 19px 'Barlow Condensed', sans-serif !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid var(--border); margin-top: 14px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 0; padding: 8px 0; height: auto; }
.stTabs [role="tab"], .stTabs [role="tab"] * { color: var(--muted); font: 500 19px 'Barlow Condensed', sans-serif !important; }
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * { color: var(--text) !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Misc widgets ---------- */
[data-testid="stExpander"] details { background: transparent; border: 0 !important; border-bottom: 1px solid var(--border) !important; border-radius: 0; }
[data-testid="stExpander"] summary { padding-left: 0; padding-right: 0; }
[data-testid="stExpander"] summary p { color: var(--text); font-weight: 400; font-size: 14px; }
[data-testid="stExpander"] summary svg { fill: var(--muted); }
[data-testid="stAlert"], [data-testid="stAlertContainer"] { background: var(--surface) !important; border-radius: 2px !important; }
[data-testid="stAlert"] { border: 1px solid var(--border); border-left: 3px solid var(--accent); }
[data-testid="stAlert"] p, [data-testid="stAlertContainer"] p { color: var(--text) !important; }
.stApp code { background: var(--bg2) !important; color: var(--text) !important; border: 1px solid var(--border); border-radius: 2px; padding: 0 5px; }
[data-testid="stTooltipIcon"] svg, [data-testid="stExpander"] summary svg { stroke: var(--muted); color: var(--muted); }
[data-testid="stSpinner"] p, [data-testid="stSpinner"] div { color: var(--text) !important; }
[data-testid="stProgress"] [role="progressbar"] > div { background: var(--border) !important; height: 3px !important; border-radius: 0 !important; }
[data-testid="stProgress"] [role="progressbar"] > div > div { background: var(--text) !important; border-radius: 0 !important; }
[data-testid="stImage"] img, video { border-radius: 0; }
[data-testid="stVideo"] { border-radius: 0; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stDeckGlJsonChart"] { border-radius: 0; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stToggle"] p, [data-testid="stCheckbox"] p, [data-testid="stRadio"] p { color: var(--text) !important; }

/* ---------- Header ---------- */
.top { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, 440px); gap: 16px 48px; align-items: end; }
.top h1 { font: 700 clamp(34px, 4.6vw, 60px)/.95 'Barlow Condensed', sans-serif !important; margin: 0 0 16px !important; padding: 0 !important; max-width: 15ch; color: var(--text); }
.top .lead { max-width: 52ch; color: var(--muted) !important; margin: 0 0 12px; font-size: 16px; }
.status { display: flex; flex-wrap: wrap; gap: 4px 22px; font-size: 13px; color: var(--muted); }
.status b { color: var(--text); font-weight: 600; }
.scene { border: 1px solid var(--border); }
.scene svg { display: block; width: 100%; height: auto; }
@media (max-width: 900px) { .top { grid-template-columns: 1fr; } }
/* Streamlit adds a link icon to every heading; hide it on our own headings */
.top h1 a, .panel-title h3 a, .step h4 a, [data-testid="stHeaderActionElements"] { display: none !important; }
.dbl { margin: 20px 0 22px; height: 7px; border-top: 2px solid var(--accent); border-bottom: 2px solid var(--accent); }

/* ---------- Strip plan ---------- */
.plan { border: 1px solid var(--border); background: var(--surface); }
.plan-h { display: flex; flex-wrap: wrap; gap: 4px 20px; justify-content: space-between; align-items: baseline; padding: 10px 14px;
  border-bottom: 1px solid var(--border); font-size: 13px; color: var(--muted); }
.plan-h b { color: var(--text); font: 600 18px 'Barlow Condensed', sans-serif; }
.plan .sc { overflow-x: auto; }
.plan svg { display: block; min-width: 760px; width: 100%; }

/* ---------- Source line ---------- */
.src { display: flex; flex-wrap: wrap; gap: 6px 26px; align-items: center; padding: 8px 0; color: var(--muted); font-size: 14px; font-variant-numeric: tabular-nums; }
.src b { color: var(--text); font-weight: 600; word-break: break-all; }

/* ---------- KPI row ---------- */
.kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); border-top: 1px solid var(--text); border-bottom: 1px solid var(--border); }
.kpi { padding: 14px 18px 14px 0; }
.kpi + .kpi { padding-left: 18px; border-left: 1px solid var(--border); }
.kpi .l { font-size: 13px; color: var(--muted); }
.kpi .fig { font: 600 52px/1.05 'Barlow Condensed', sans-serif; font-variant-numeric: tabular-nums; }
.kpi .fig small { font-size: 18px; color: var(--muted); margin-left: 4px; font-weight: 500; }
.kpi .s { font-size: 12.5px; color: var(--faint); }
@media (max-width: 1100px) { .kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } .kpi { padding-left: 0 !important; border-left: 0 !important; } }
@media (max-width: 860px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .status { text-align: left; } }

/* ---------- Status & panels ---------- */
.stat { font-size: 13px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 24px; font-variant-numeric: tabular-nums; }
.stat b { color: var(--text); }
.panel-title { margin: 14px 0 10px; }
.panel-title h3 span { display: inline; font: inherit; color: inherit; }
.panel-title h3 { font: 600 22px/1.2 'Barlow Condensed', sans-serif !important; margin: 0 0 2px !important; padding: 0 !important; color: var(--text); }
.panel-title.big h3 { font-size: 30px !important; }
.panel-title > span { display: block; font-size: 13px; color: var(--muted); }
.panel-title .live { color: var(--pothole); font-weight: 600; }
.placeholder { display: grid; place-items: center; text-align: center; height: 360px; border: 1px solid var(--border);
  background: var(--surface); color: var(--muted); font-size: 14px; }

/* ---------- Empty state ---------- */
.steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid var(--text); border-bottom: 1px solid var(--border); }
.step { padding: 16px 18px 16px 0; }
.step + .step { padding-left: 18px; border-left: 1px solid var(--border); }
.step .n { font: 600 34px/1 'Barlow Condensed', sans-serif; color: var(--accent); }
.step h4 { margin: 8px 0 4px !important; font-size: 20px !important; padding: 0 !important; }
.step p { margin: 0; color: var(--muted) !important; font-size: 14px; }
@media (max-width: 900px) { .steps { grid-template-columns: 1fr; } .step { padding-left: 0 !important; border-left: 0 !important; border-top: 1px solid var(--border); } }

/* ---------- Stat tiles ---------- */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0 18px; border-top: 1px solid var(--text); }
.tile { padding: 10px 0; border-bottom: 1px solid var(--border); }
.tile .l { font-size: 13px; color: var(--muted); }
.tile .v { font: 600 26px/1.15 'Barlow Condensed', sans-serif; color: var(--text); font-variant-numeric: tabular-nums; }

/* ---------- Registry table ---------- */
.reg-wrap { max-height: 560px; overflow: auto; }
.reg { width: 100%; border-collapse: collapse; font-size: 14px; }
.reg th { position: sticky; top: 0; background: var(--bg); color: var(--muted); text-align: left; font-size: 13px; font-weight: 500;
  padding: 6px 12px 6px 0; border-bottom: 1px solid var(--text); z-index: 1; }
.reg td { padding: 8px 12px 8px 0; border-bottom: 1px solid var(--border); color: var(--text); white-space: nowrap; }
.pill { color: var(--text); white-space: nowrap; }
.pill::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 7px; background: var(--c); }
.bar { display: inline-block; vertical-align: middle; height: 3px; width: 70px; background: var(--border); margin-right: 8px; }
.bar > span { display: block; height: 100%; background: var(--text); }

/* ---------- Legend, priority mix, evidence ---------- */
.legend { display: flex; flex-direction: column; border-top: 1px solid var(--text); }
.legend-row { display: flex; justify-content: space-between; align-items: center; font-size: 14px; color: var(--text); padding: 7px 0; border-bottom: 1px solid var(--border); }
.legend-row .sw { display: inline-block; width: 9px; height: 9px; background: var(--c); margin-right: 7px; }
.mix { display: flex; height: 14px; margin: 6px 0 8px; }
.mix i { display: block; }
.note { font-size: 13px; color: var(--muted); }
.ev-cap { display: flex; justify-content: space-between; align-items: center; margin: 4px 0 16px; font-size: 13px; color: var(--muted); }
.ev-cap b { color: var(--text); }
"""


def inject_theme(T: dict):
    vars_css = ";".join(f"--{k}:{v}" for k, v in T.items() if not k in ("shadow", "map")) + f";--shadow:{T['shadow']}"
    st.markdown(f"<style>{FONT_IMPORT}:root{{{vars_css}}}{BASE_CSS}</style>", unsafe_allow_html=True)


def sync_streamlit_theme(T: dict, dark: bool):
    """Point Streamlit's own theme at our palette so built-in widgets (dropdowns, alerts, code) match."""
    want = {
        "theme.base": "dark" if dark else "light",
        "theme.primaryColor": T["accent"],
        "theme.backgroundColor": T["bg"],
        "theme.secondaryBackgroundColor": T["surface"],
        "theme.textColor": T["text"],
    }
    if any(st.get_option(k) != v for k, v in want.items()):
        for k, v in want.items():
            st._config.set_option(k, v)  # no public API for switching theme at runtime
        st.rerun()


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def class_color(cls: str, T: dict) -> str:
    return T.get(str(cls).lower(), T["ok"])


def priority_color(p: str, T: dict) -> str:
    return {"High": T["pothole"], "Medium": T["accent"], "Low": T["ok"]}.get(p, T["muted"])


def hex_to_rgba(h: str, a: int = 230):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [a]


def style_chart(chart, T: dict):
    return (
        chart.configure(background="transparent", font="Barlow")
        .configure_axis(
            labelColor=T["muted"], titleColor=T["muted"], gridColor=T["grid"], gridOpacity=0.6,
            domainColor=T["text"], tickColor=T["border"], labelFontSize=12, titleFontSize=12, titleFontWeight="normal",
        )
        .configure_legend(labelColor=T["text"], titleColor=T["muted"], orient="top", labelFontSize=12, symbolType="square")
        .configure_view(strokeWidth=0)
    )


# ==================================================
# Helpers: model, video, geometry
# ==================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    return YOLO(weights_path)


def find_available_models():
    models = []
    candidates = [
        PROJECT_DIR / "best.pt",
        PROJECT_DIR / "runs" / "road_damage_baseline" / "weights" / "best_640.pt",
        PROJECT_DIR / "runs" / "road_damage_2class_640" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.is_file() and str(c) not in models:
            models.append(str(c))
    for extra in PROJECT_DIR.glob("*.pt"):
        if str(extra) not in models and extra.name != PRIVACY_MODEL:  # COCO privacy model is not a hazard model
            models.append(str(extra))
    return models if models else ["best.pt"]


def normalize_hazard_class(raw_name: str) -> str:
    """Normalizes class names across RDD2022 (D00, D10, D20, D40) and custom labels."""
    name = str(raw_name).lower().strip()
    if any(k in name for k in ["pothole", "d40", "rutting", "bump"]):
        return "pothole"
    if any(k in name for k in ["manhole", "d70", "drain", "utility"]):
        return "manhole"
    if any(k in name for k in ["crack", "d00", "d10", "d20", "fissure"]):
        return "crack"
    return name


def find_test_videos():
    tests_dir = PROJECT_DIR / "tests"
    if not tests_dir.is_dir():
        return []
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [p for p in sorted(tests_dir.iterdir()) if p.suffix.lower() in valid_exts]


@st.cache_data(show_spinner=False)
def probe_video(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "duration": n / fps if fps else 0}


def convert_to_web_mp4(input_video_path, output_video_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", str(input_video_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_video_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def offset_coords(lat0, lon0, dist_m, bearing_deg):
    b = np.radians(bearing_deg)
    dlat = dist_m * np.cos(b) / 111111.0
    dlon = dist_m * np.sin(b) / (111111.0 * np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def priority_of(cls: str, conf: float, area_ratio: float):
    """Repair priority from detector confidence and apparent size in frame."""
    size = min(1.0, area_ratio / 0.02)  # a box covering ~2% of the frame counts as 'large'
    score = 0.55 * conf + 0.45 * size
    if cls == "manhole":
        score *= 0.8  # utility asset rather than surface failure
    label = "High" if score >= 0.6 else "Medium" if score >= 0.38 else "Low"
    return score, label


def fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def draw_hazard(img, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = y1 - 8 if y1 - th - 10 > 0 else y2 + th + 10
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), color, -1)
    cv2.putText(img, text, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


def draw_hud(img, width, frame_idx, total_frames, cum):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, 46), (19, 14, 10), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (0, 46), (int(width * frame_idx / max(1, total_frames)), 49), (5, 183, 242), -1)
    left = f"POTHOLES {cum.get('pothole', 0)}   MANHOLES {cum.get('manhole', 0)}   TOTAL {sum(cum.values())}"
    cv2.putText(img, left, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    right = f"FRAME {frame_idx}/{total_frames}"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, right, (width - tw - 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 200, 210), 1, cv2.LINE_AA)


# ==================================================
# UI fragments
# ==================================================
def kpi_html(label, value, sub, color_var="--text", unit=""):
    unit_html = f"<small>{unit}</small>" if unit else ""
    return (
        f'<div class="kpi"><div class="l">{label}</div>'
        f'<div class="fig" style="color:var({color_var})">{value}{unit_html}</div>'
        f'<div class="s">{sub}</div></div>'
    )


def render_kpis(slot, cum, in_frame, density, fps_val, fps_sub):
    """Renders the five KPI figures as one ruled row into a single placeholder."""
    cells = [
        kpi_html("Unique potholes", cum.get("pothole", 0), f"In frame now: {in_frame.get('pothole', 0)}", "--pothole"),
        kpi_html("Unique manholes", cum.get("manhole", 0), f"In frame now: {in_frame.get('manhole', 0)}", "--manhole"),
        kpi_html("Total hazards", sum(cum.values()), "Kalman de-duplicated"),
        kpi_html("Hazard density", f"{density:.1f}", "per km of surveyed road", unit="per km"),
        kpi_html("Processing speed", f"{fps_val:.1f}", fps_sub, unit="fps"),
    ]
    slot.markdown(f'<div class="kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def strip_plan_html(hazards, lane, route_m, cursor_m=None, cursor_label=""):
    """Top-down strip plan of the road: hazards placed by chainage (x) and position across the frame (y).

    hazards: dicts with id, cls, chainage_m, score (and priority, if known, for the labels).
    lane: hazard id -> 0..1 horizontal position in the frame. cursor_m: progress marker while a run is live.
    """
    route_m = max(1.0, route_m)
    X0, X1 = 40, 1180

    def sx(m):
        return X0 + (X1 - X0) * min(1.0, max(0.0, m / route_m))

    step = next(s for s in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 1e9) if route_m / s <= 8)
    marks_m = np.arange(0, route_m - step * 0.4, step)
    ticks = "".join(f'<path d="M{sx(m):.0f} 28v12"/>' for m in marks_m) + f'<path d="M{X1} 28v12"/>'
    nums = "".join(f'<text x="{sx(m):.0f}" y="20">{m:.0f}</text>' for m in marks_m) + f'<text x="{X1}" y="20">{route_m:.0f} m</text>'

    dots, lines, labels = [], [], []
    for h in hazards:
        x, y = sx(h["chainage_m"]), 90 + 44 * min(1.0, max(0.0, lane.get(h["id"], 0.5)))
        if h["cls"] == "manhole":
            dots.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="7" fill="var(--road)" stroke="var(--manhole)" stroke-width="3"/>')
        else:
            rx = 5 + 8 * h["score"]
            col = "var(--crack)" if h["cls"] == "crack" else "var(--pothole)"
            dots.append(f'<ellipse cx="{x:.0f}" cy="{y:.0f}" rx="{rx:.1f}" ry="{rx * 0.6:.1f}" fill="{col}" stroke="#F4F2EC" stroke-width="1.2"/>')
    # Call out the three most urgent hazards, staggered so neighbouring labels don't collide
    for i, h in enumerate(sorted(sorted(hazards, key=lambda h: h["score"], reverse=True)[:3], key=lambda h: h["chainage_m"])):
        x, y = sx(h["chainage_m"]), 90 + 44 * min(1.0, max(0.0, lane.get(h["id"], 0.5)))
        ty = 176 + 20 * (i % 2)
        anchor = "start" if x < 120 else "end" if x > 1100 else "middle"
        prio = f', {h["priority"].lower()}' if h.get("priority") else ""
        lines.append(f'<path d="M{x:.0f} {y + 8:.0f}V{ty - 14}"/>')
        labels.append(f'<text x="{x:.0f}" y="{ty}" text-anchor="{anchor}">{h["cls"].capitalize()} {h["id"]}{prio}</text>')

    cursor = ""
    if cursor_m is not None:
        cx = sx(cursor_m)
        side = f'x="{cx - 14:.0f}" text-anchor="end"' if cx > 900 else f'x="{cx + 14:.0f}"'
        cursor = (
            f'<rect x="{cx:.0f}" y="76" width="{X1 - cx:.0f}" height="72" fill="var(--bg)" opacity=".6"/>'
            f'<path d="M{cx:.0f} 48V152" stroke="var(--text)" stroke-width="2.5"/>'
            f'<path d="M{cx - 8:.0f} 48h16l-8 10z" fill="var(--text)"/>'
            f'<text {side} y="62" font-size="13" font-weight="600" fill="var(--text)">{cursor_label}</text>'
        )
    return (
        '<section class="plan"><div class="plan-h"><b>Strip plan of the surveyed road</b>'
        f'<span class="num">Chainage 0 to {route_m:.0f} m, driving left to right · {len(hazards)} hazards</span></div>'
        '<div class="sc"><svg viewBox="0 0 1220 210" role="img" aria-label="Strip plan: hazards plotted along the surveyed road by chainage" font-family="Barlow, sans-serif">'
        f'<g stroke="var(--faint)" stroke-width="1"><path d="M{X0} 34H{X1}"/>{ticks}</g>'
        f'<g font-size="13" fill="var(--muted)" text-anchor="middle">{nums}</g>'
        f'<rect x="{X0}" y="80" width="{X1 - X0}" height="64" fill="var(--road)"/>'
        f'<path d="M{X0} 83H{X1}M{X0} 141H{X1}" stroke="#F4F2EC" stroke-width="1.5" opacity=".75"/>'
        f'<path d="M{X0} 112H{X1}" stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="26 22"/>'
        f'{"".join(dots)}<g stroke="var(--text)" stroke-width="1">{"".join(lines)}</g>'
        f'<g font-size="13" fill="var(--text)">{"".join(labels)}</g>{cursor}</svg></div></section>'
    )


def timeline_chart(history, T, fps):
    df = pd.DataFrame(history)
    if len(df) > 600:
        df = df.iloc[:: max(1, len(df) // 600)]
    df["Time (s)"] = df["Frame"] / fps
    long = df.melt(id_vars=["Frame", "Time (s)"], value_vars=["Potholes", "Manholes"], var_name="Class", value_name="Count")
    color = alt.Color("Class:N", scale=alt.Scale(domain=["Potholes", "Manholes"], range=[T["pothole"], T["manhole"]]), legend=alt.Legend(title=None))
    base = alt.Chart(long).encode(
        x=alt.X("Time (s):Q", title="Video time (s)"),
        y=alt.Y("Count:Q", title="Unique hazards", axis=alt.Axis(tickMinStep=1)),
        color=color,
    )
    return style_chart(base.mark_line(strokeWidth=3, interpolate="step-after").properties(height=320), T)


def registry_table_html(records, T):
    head = "".join(f"<th>{h}</th>" for h in ["ID", "Type", "Priority", "Chainage", "Video time", "Tracked", "Peak confidence", "Size", "GPS"])
    rows = []
    for r in records:
        cc = class_color(r["cls"], T)
        pc = priority_color(r["priority"], T)
        rows.append(
            "<tr>"
            f'<td class="mono">{r["id"]}</td>'
            f'<td><span class="pill" style="--c:{cc}">{r["cls"].capitalize()}</span></td>'
            f'<td><span class="pill" style="--c:{pc}">{r["priority"]}</span></td>'
            f'<td class="mono">{r["chainage_m"] / 1000:.3f} km</td>'
            f'<td class="mono">{r["time_s"]:.1f}s</td>'
            f'<td class="mono">{r["frames"]} fr</td>'
            f'<td><span class="bar"><span style="width:{r["conf"] * 100:.0f}%;--c:{cc}"></span></span><span class="mono">{r["conf"]:.0%}</span></td>'
            f'<td class="mono">{r["area_pct"]:.2f}%</td>'
            f'<td class="mono">{r["lat"]:.5f}, {r["lon"]:.5f}</td>'
            "</tr>"
        )
    return f'<div class="reg-wrap"><table class="reg"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def grouped_bars(rows, columns, T, colors, fmt="%"):
    """Small grouped bar chart: rows of (group, series, value)."""
    df = pd.DataFrame(rows, columns=columns)
    group, series, value = columns
    order = list(dict.fromkeys(df[group]))
    names = list(dict.fromkeys(df[series]))
    axis = alt.Axis(format=fmt, tickCount=5) if fmt else alt.Axis(tickCount=5)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{group}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset(f"{series}:N", sort=names),
        y=alt.Y(f"{value}:Q", title=None, axis=axis),
        color=alt.Color(f"{series}:N", sort=names, scale=alt.Scale(domain=names, range=colors), legend=alt.Legend(title=None)),
        tooltip=[group, series, alt.Tooltip(f"{value}:Q", format=".1%" if fmt else ",")],
    ).properties(height=250)
    return style_chart(chart, T)


def render_model_overview(T):
    """Charts that are always available: how accurate the baseline model is and what it was trained on."""
    html('<div class="panel-title big"><h3>Model and data at a glance</h3>'
         '<span>Baseline YOLO11n, measured on images it never saw in training.</span></div>')
    class_cols = [T["pothole"], T["manhole"]]
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        html('<div class="panel-title"><h3>Accuracy by hazard type</h3><span>Manholes are found more reliably than potholes.</span></div>')
        st.altair_chart(grouped_bars(BASELINE_ACCURACY, ["Metric", "Class", "Score"], T, class_cols), use_container_width=True, theme=None)
    with c2:
        html('<div class="panel-title"><h3>Training data</h3><span>Labelled objects in each split of the dataset.</span></div>')
        st.altair_chart(grouped_bars(DATASET_OBJECTS, ["Split", "Class", "Objects"], T, class_cols, fmt=""), use_container_width=True, theme=None)
    with c3:
        html('<div class="panel-title"><h3>Image size tested</h3><span>640 px scored slightly higher than 768 px on validation.</span></div>')
        st.altair_chart(grouped_bars(RESOLUTION_VAL, ["Metric", "Image size", "Score"], T, [T["text"], T["accent"]]), use_container_width=True, theme=None)


def render_empty_state():
    html(
        '<div class="steps">'
        '<div class="step"><div class="n">01</div><h4>Load footage</h4><p>Upload dash-cam video or pick a clip from the <code>tests/</code> folder in the sidebar.</p></div>'
        '<div class="step"><div class="n">02</div><h4>Detect &amp; track</h4><p>YOLO detects hazards each frame; the Kalman tracker assigns stable IDs so each pothole is counted once.</p></div>'
        '<div class="step"><div class="n">03</div><h4>Map, prioritise, export</h4><p>Get a geo-referenced hazard map, repair priorities, evidence snapshots and CSV / JSON / GeoJSON exports.</p></div>'
        "</div>"
    )


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    html('<div class="brand">RoadSense AI<small>Road hazard inspection</small></div>')
    dark = st.toggle("Dark theme", value=False, key="dark_mode")

T = THEMES["dark" if dark else "light"]
sync_streamlit_theme(T, dark)
inject_theme(T)

with st.sidebar:
    html('<div class="side-sec">Model</div>')
    available_models = find_available_models()
    selected_model_path = st.selectbox("Model weights", options=available_models, index=0, format_func=lambda p: Path(p).name)

    try:
        with st.spinner("Loading model…"):
            model = load_yolo_model(selected_model_path)
        model_classes = {k: v.lower() for k, v in model.names.items()}
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    gpu = torch.cuda.is_available()
    device = st.selectbox(
        "Compute device",
        options=["0", "cpu"] if gpu else ["cpu"],
        format_func=lambda d: f"GPU · {torch.cuda.get_device_name(0)}" if d == "0" else "CPU",
    )
    html(f'<div class="row"><span>Classes</span><span>{", ".join(model_classes.values())}</span></div>')

    html('<div class="side-sec">Detection</div>')
    conf_thresh = st.slider("Confidence", 0.05, 1.0, 0.15, 0.05,
                            help="0.15 recommended for high recall on subtle or shadowed potholes.")
    use_tta = st.toggle("Test-time augmentation", value=True,
                        help="Multi-scale flipped ensembling. Better recall on small/distant potholes, ~2-3x slower.")
    anonymize_on = st.toggle("Blur people and vehicles", value=True,
                             help="Privacy: blurs faces and number plates before frames are shown, saved or exported.")
    privacy_model = None
    if anonymize_on:
        try:
            privacy_model = load_yolo_model(PRIVACY_MODEL)
        except Exception as e:
            st.warning(f"Privacy model unavailable ({e}). Output will NOT be anonymized.")

    with st.expander("Tracking and performance"):
        iou_thresh = st.slider("IoU matching threshold", 0.1, 0.9, 0.25, 0.05,
                               help="Spatial IoU threshold for Hungarian matching.")
        min_confirm_hits = st.slider("Anti-flicker confirmation frames", 1, 5, 2,
                                     help="Consecutive frames a hazard must appear before it is counted.")
        frame_stride = st.select_slider("Frame stride", options=[1, 2, 3, 4], value=1,
                                        help="1 = every frame. 2 = every 2nd frame (2x faster).")
        ui_every = st.select_slider("Live preview refresh", options=[1, 2, 3, 5, 10], value=2,
                                    help="Update the dashboard every N processed frames. Higher = faster analysis.")

    with st.expander("Geo-reference"):
        base_lat = st.number_input("Start latitude", value=-28.753000, format="%.6f")
        base_lon = st.number_input("Start longitude", value=31.895000, format="%.6f")
        bearing = st.slider("Route bearing (°)", 0, 359, 45, help="Direction of travel, 0 = north, 90 = east.")
        speed_kmh = st.slider("Average vehicle speed (km/h)", 10, 120, 45)

    html('<div class="side-sec">Video</div>')
    input_mode = st.radio("Input mode", ["Upload", "tests/ folder"], horizontal=True, label_visibility="collapsed")

    video_input_path = None
    input_filename = None

    if input_mode == "Upload":
        uploaded_file = st.file_uploader("Upload road video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if uploaded_file is not None:
            up_key = f"{uploaded_file.name}-{uploaded_file.size}"
            # Write to disk once per file, not on every rerun
            if st.session_state.get("upload_key") != up_key:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
                tfile.write(uploaded_file.getbuffer())
                tfile.close()
                st.session_state["upload_key"] = up_key
                st.session_state["upload_path"] = tfile.name
            video_input_path = st.session_state["upload_path"]
            input_filename = uploaded_file.name
    else:
        sample_videos = find_test_videos()
        if sample_videos:
            chosen_video = st.selectbox("Sample video", options=sample_videos, format_func=lambda p: p.name)
            if chosen_video:
                video_input_path = str(chosen_video)
                input_filename = chosen_video.name
        else:
            st.warning("No sample videos found in `tests/`.")

# ==================================================
# Hero header
# ==================================================
device_label = "GPU" if device != "cpu" else "CPU"
html(
    '<div class="top"><div><h1>Road Damage Detection &amp; Spatial Inspection</h1>'
    '<p class="lead">Dash-cam footage goes in. A ranked, mapped list of potholes and manholes comes out, each counted once.</p>'
    f'<div class="status"><span><b>Model ready</b></span><span>{Path(selected_model_path).name} on {device_label}</span>'
    f'<span>Test-time augmentation {"on" if use_tta else "off"}</span>'
    f'<span>Privacy blur {"on" if privacy_model else "off"}</span></div></div>'
    f'<div class="scene">{HERO_SCENE_SVG}</div></div><div class="dbl"></div>'
)

if not video_input_path:
    render_empty_state()
    st.write("")
    st.info("Upload a road video or choose a sample in the sidebar to begin.")
    st.write("")
    render_model_overview(T)
    st.stop()

meta = probe_video(video_input_path)
results = st.session_state.get("results")
if results and results.get("source") != input_filename:
    results = None

# Strip plan of the road (empty before a run, live during one, final afterwards)
plan_slot = st.empty()
est_route_m = meta["duration"] * speed_kmh / 3.6
if results:
    plan_slot.markdown(strip_plan_html(results["records"], results.get("lane", {}), results["route_m"]), unsafe_allow_html=True)
else:
    plan_slot.markdown(strip_plan_html([], {}, est_route_m), unsafe_allow_html=True)

# Source line + run button
src_col, btn_col = st.columns([5, 1], vertical_alignment="center")
with src_col:
    html(
        f'<div class="src"><span><b>{input_filename}</b></span><span>{meta["w"]}×{meta["h"]}</span>'
        f'<span>{meta["fps"]:.0f} fps</span><span>{meta["frames"]} frames</span><span>{meta["duration"]:.1f} s</span>'
        f'<span>{est_route_m / 1000:.2f} km at {speed_kmh} km/h</span></div>'
    )
with btn_col:
    start_btn = st.button("Run inspection", type="primary", use_container_width=True)

# KPI row
kpi_slot = st.empty()
if results:
    render_kpis(kpi_slot, results["cumulative"], {}, results["density"], results["avg_fps"], "average over the run")
else:
    render_kpis(kpi_slot, {}, {}, 0.0, 0.0, "awaiting run")

progress_bar = st.progress(1.0 if results else 0.0)
status_slot = st.empty()
if results:
    status_slot.markdown(
        f'<div class="stat"><span><b>Analysis complete</b></span><span>Hazards {results["total"]}</span>'
        f'<span>Run time {fmt_dur(results["elapsed"])}</span><span>Average speed {results["avg_fps"]:.1f} fps</span></div>',
        unsafe_allow_html=True,
    )

st.write("")
feed_col, chart_col = st.columns([6, 5], gap="large")
with feed_col:
    feed_title = st.empty()
    live_frame_view = st.empty()
with chart_col:
    html('<div class="panel-title"><h3>Cumulative detections</h3><span>Hazards counted so far, by video time.</span></div>')
    live_chart_view = st.empty()

FEED_CAPTION = ("People and vehicles are blurred before anything is saved." if privacy_model
                else "Privacy blur is off: people and vehicles are not blurred.")
if not start_btn:
    feed_title.markdown(f'<div class="panel-title"><h3>Live inspection feed</h3><span>{FEED_CAPTION}</span></div>', unsafe_allow_html=True)
    if results and results.get("last_frame") is not None:
        live_frame_view.image(results["last_frame"], use_container_width=True)
    else:
        live_frame_view.markdown('<div class="placeholder"><div>Press <b>Run inspection</b> to start the live feed</div></div>', unsafe_allow_html=True)
    if results and results["history"]:
        live_chart_view.altair_chart(timeline_chart(results["history"], T, results["fps"]), use_container_width=True, theme=None)
    else:
        live_chart_view.markdown('<div class="placeholder" style="height:320px"><div>The timeline appears once analysis starts</div></div>', unsafe_allow_html=True)

# ==================================================
# Analysis loop
# ==================================================
if start_btn:
    st.session_state.pop("results", None)
    feed_title.markdown(f'<div class="panel-title"><h3>Live inspection feed</h3><span><span class="live" style="display:inline">Live.</span> {FEED_CAPTION}</span></div>', unsafe_allow_html=True)

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(max(1, width * height))
    m_per_frame = (speed_kmh / 3.6) / orig_fps  # metres travelled per raw video frame

    temp_dir = tempfile.mkdtemp()
    temp_raw_video = Path(temp_dir) / "annotated_raw.avi"
    final_output_mp4 = Path(temp_dir) / "road_damage_tracked.mp4"

    effective_fps = max(1.0, orig_fps / frame_stride)
    out_writer = cv2.VideoWriter(str(temp_raw_video), cv2.VideoWriter_fourcc(*"MJPG"), effective_fps, (width, height))

    tracker = RoadHazardKalmanTracker(
        min_confirmation_hits=min_confirm_hits,
        max_age=int(orig_fps * 1.5),
        iou_threshold=iou_thresh,
    )

    frame_idx = 0
    processed = 0
    t_start = time.time()
    last_t = t_start
    current_fps = 0.0
    history = []
    peak_area = {}  # hazard id -> largest bbox area ratio seen
    lane = {}  # hazard id -> 0..1 horizontal position in the frame (for the strip plan)
    evidence = {}  # hazard id -> (confidence, jpeg bytes)
    cumulative = {}
    preview_rgb = None
    disp_scale = min(1.0, 960 / max(1, width))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue
        processed += 1

        res = model.predict(source=frame, conf=conf_thresh, iou=iou_thresh, device=device, augment=use_tta, verbose=False)[0]
        detections = [
            {
                "bbox": box.xyxy[0].tolist(),
                "class_name": normalize_hazard_class(model.names.get(int(box.cls[0].item()), "")),
                "conf": float(box.conf[0].item()),
            }
            for box in res.boxes
        ]

        visible_hazards, cumulative = tracker.update(detections, frame_idx)

        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            inst = 1.0 / dt
            current_fps = inst if current_fps == 0 else 0.85 * current_fps + 0.15 * inst

        # Hazards are detected on the raw frame; everything shown or saved comes from the blurred one
        safe = anonymize(frame, privacy_model, device) if privacy_model else frame
        annotated = safe.copy()
        in_frame = {}
        for hz in visible_hazards:
            x1, y1, x2, y2 = [int(v) for v in hz.get_bbox()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            in_frame[hz.class_name] = in_frame.get(hz.class_name, 0) + 1

            area = (x2 - x1) * (y2 - y1) / frame_area
            peak_area[hz.id] = max(peak_area.get(hz.id, 0.0), area)
            lane[hz.id] = (x1 + x2) / 2 / width

            # Keep the highest-confidence crop of each hazard as photo evidence
            conf_now = float(hz.conf)
            if conf_now >= evidence.get(hz.id, (-1.0, None))[0]:
                pw, ph = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
                crop = safe[max(0, y1 - ph) : min(height, y2 + ph), max(0, x1 - pw) : min(width, x2 + pw)]
                if crop.size:
                    s = 320 / max(crop.shape[:2])
                    if s < 1:
                        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if ok:
                        evidence[hz.id] = (conf_now, buf.tobytes())

            draw_hazard(annotated, (x1, y1, x2, y2), DRAW_BGR.get(hz.class_name, (94, 197, 34)),
                        f"{hz.class_name} {hz.id}  {conf_now:.2f}")

        draw_hud(annotated, width, frame_idx, total_frames, cumulative)
        out_writer.write(annotated)

        history.append({"Frame": frame_idx, "Potholes": cumulative.get("pothole", 0), "Manholes": cumulative.get("manhole", 0)})

        if processed % ui_every == 0 or processed == 1:
            surveyed_km = frame_idx * m_per_frame / 1000
            density = sum(cumulative.values()) / surveyed_km if surveyed_km > 0.01 else 0.0
            render_kpis(kpi_slot, cumulative, in_frame, density, current_fps, f"frame {frame_idx} of {total_frames}")

            pct = min(1.0, frame_idx / total_frames)
            progress_bar.progress(pct)
            elapsed = now - t_start
            eta = elapsed / pct * (1 - pct) if pct > 0 else 0
            status_slot.markdown(
                f'<div class="stat"><span><b>Processing</b> {pct:.0%}</span>'
                f'<span>Frame {frame_idx}/{total_frames}</span><span>Hazards {sum(cumulative.values())}</span>'
                f'<span>Elapsed {fmt_dur(elapsed)}</span><span>ETA {fmt_dur(eta)}</span></div>',
                unsafe_allow_html=True,
            )

            live_hazards = [
                {"id": hid, "cls": info["class_name"], "chainage_m": info["first_frame"] * m_per_frame,
                 "score": priority_of(info["class_name"], float(info["peak_conf"]), peak_area.get(hid, 0.0))[0]}
                for hid, info in tracker.hazard_registry.items()
            ]
            plan_slot.markdown(
                strip_plan_html(live_hazards, lane, total_frames * m_per_frame, frame_idx * m_per_frame,
                                f"Frame {frame_idx} of {total_frames}, {frame_idx * m_per_frame:.0f} m"),
                unsafe_allow_html=True,
            )

            disp = cv2.resize(annotated, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA) if disp_scale < 1 else annotated
            preview_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            live_frame_view.image(preview_rgb, use_container_width=True)

        if processed % 15 == 0:
            live_chart_view.altair_chart(timeline_chart(history, T, orig_fps), use_container_width=True, theme=None)

    cap.release()
    out_writer.release()
    elapsed = time.time() - t_start

    if history:
        live_chart_view.altair_chart(timeline_chart(history, T, orig_fps), use_container_width=True, theme=None)
    progress_bar.progress(1.0)

    with st.spinner("Encoding browser-ready video…"):
        try:
            convert_to_web_mp4(temp_raw_video, final_output_mp4)
            output_video_path = str(final_output_mp4)
        except Exception as err:
            st.warning(f"MP4 encoding failed, falling back to raw AVI: {err}")
            output_video_path = str(temp_raw_video)

    # Build geo-referenced hazard registry
    records = []
    for hid, info in sorted(tracker.hazard_registry.items(), key=lambda kv: kv[1]["first_frame"]):
        cls = info["class_name"]
        chain_m = info["first_frame"] * m_per_frame
        lat, lon = offset_coords(base_lat, base_lon, chain_m, bearing)
        area = peak_area.get(hid, 0.0)
        score, prio = priority_of(cls, float(info["peak_conf"]), area)
        records.append(
            {
                "id": hid,
                "cls": cls,
                "first_frame": int(info["first_frame"]),
                "last_frame": int(info["last_frame"]),
                "frames": int(info["last_frame"] - info["first_frame"] + 1),
                "time_s": round(info["first_frame"] / orig_fps, 2),
                "conf": float(info["peak_conf"]),
                "area_pct": round(area * 100, 3),
                "score": round(score, 3),
                "priority": prio,
                "chainage_m": round(chain_m, 1),
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    route_m = total_frames * m_per_frame
    cum_final = dict(tracker.cumulative_counts)
    total_unique = sum(cum_final.values())
    avg_fps = processed / elapsed if elapsed > 0 else 0.0
    density = total_unique / (route_m / 1000) if route_m > 10 else 0.0

    st.session_state["results"] = {
        "source": input_filename,
        "video_path": output_video_path,
        "records": records,
        "evidence": {hid: v[1] for hid, v in evidence.items()},
        "cumulative": cum_final,
        "total": total_unique,
        "history": history,
        "fps": orig_fps,
        "route_m": route_m,
        "density": density,
        "elapsed": elapsed,
        "avg_fps": avg_fps,
        "processed": processed,
        "resolution": f"{width}×{height}",
        "last_frame": preview_rgb,
        "lane": lane,
        "settings": {
            "model": Path(selected_model_path).name, "conf": conf_thresh, "tta": use_tta,
            "anonymized": privacy_model is not None,
            "iou": iou_thresh, "confirm": min_confirm_hits, "stride": frame_stride,
            "speed_kmh": speed_kmh, "bearing": bearing, "base": [base_lat, base_lon],
        },
    }
    st.rerun()

# ==================================================
# Results
# ==================================================
if results and Path(results["video_path"]).is_file():
    records = results["records"]
    st.write("")
    tab_video, tab_map, tab_analytics, tab_registry, tab_evidence = st.tabs(
        ["Video", "Map", "Analytics", "Registry and export", "Evidence"]
    )
    stem = Path(results["source"]).stem
    prio_counts = {p: sum(1 for r in records if r["priority"] == p) for p in PRIORITY_ORDER}

    # ---------------- Video ----------------
    with tab_video:
        v_col, s_col = st.columns([3, 2], gap="large")
        with v_col:
            with open(results["video_path"], "rb") as vf:
                v_bytes = vf.read()
            st.video(v_bytes)
            st.download_button(
                "Download annotated video", data=v_bytes, file_name=f"inspected_{stem}.mp4",
                mime="video/mp4", type="primary", use_container_width=True,
            )
        with s_col:
            html('<div class="panel-title"><h3>Inspection summary</h3><span>Each hazard is counted once.</span></div>')
            cum = results["cumulative"]
            rows = "".join(
                f'<div class="legend-row"><span><span class="sw" style="--c:{c}"></span>{lbl}</span><b class="mono">{n}</b></div>'
                for lbl, c, n in [
                    ("Potholes", T["pothole"], cum.get("pothole", 0)),
                    ("Manholes", T["manhole"], cum.get("manhole", 0)),
                    ("High priority", priority_color("High", T), prio_counts["High"]),
                    ("Medium priority", priority_color("Medium", T), prio_counts["Medium"]),
                    ("Low priority", priority_color("Low", T), prio_counts["Low"]),
                ]
            )
            html(f'<div class="legend">{rows}<div class="legend-row">'
                 f'<span><b>Total unique hazards</b></span><b class="mono">{results["total"]}</b></div></div>')
            st.write("")
            s = results["settings"]
            html(
                '<div class="tiles">'
                f'<div class="tile"><div class="l">Route</div><div class="v">{results["route_m"] / 1000:.2f} km</div></div>'
                f'<div class="tile"><div class="l">Run time</div><div class="v">{fmt_dur(results["elapsed"])}</div></div>'
                f'<div class="tile"><div class="l">Resolution</div><div class="v">{results["resolution"]}</div></div>'
                f'<div class="tile"><div class="l">Conf · TTA</div><div class="v">{s["conf"]:.2f} · {"on" if s["tta"] else "off"}</div></div>'
                "</div>"
            )

    # ---------------- Map ----------------
    with tab_map:
        if records:
            map_df = pd.DataFrame(records)
            map_df["color"] = map_df["cls"].apply(lambda c: hex_to_rgba(class_color(c, T)))
            map_df["radius"] = 6 + map_df["score"] * 16
            map_df["label"] = map_df["cls"].str.capitalize()
            map_df["conf_txt"] = (map_df["conf"] * 100).round(0).astype(int).astype(str) + "%"
            map_df["km"] = (map_df["chainage_m"] / 1000).round(3)

            s = results["settings"]
            n_pts = 60
            route_pts = [list(offset_coords(s["base"][0], s["base"][1], d, s["bearing"]))[::-1]
                         for d in np.linspace(0, results["route_m"], n_pts)]

            m_col, l_col = st.columns([3, 1], gap="large")
            with l_col:
                n_prio = max(1, len(records))
                html(
                    f'<div class="panel-title"><h3>Repair priority</h3><span>{prio_counts["High"]} high, '
                    f'{prio_counts["Medium"]} medium, {prio_counts["Low"]} low.</span></div><div class="mix">'
                    + "".join(f'<i style="width:{prio_counts[p] / n_prio:.1%};background:{priority_color(p, T)}"></i>' for p in PRIORITY_ORDER)
                    + "</div>"
                )
                html('<div class="panel-title"><h3>Layers</h3></div>')
                show_heat = st.toggle("Density heatmap", value=False)
                show_route = st.toggle("Survey route", value=True)
                classes_on_map = st.multiselect("Show classes", sorted(map_df["cls"].unique()),
                                                default=sorted(map_df["cls"].unique()), format_func=str.capitalize)
                rows = "".join(
                    f'<div class="legend-row"><span><span class="sw" style="--c:{class_color(c, T)}"></span>{c.capitalize()}</span>'
                    f'<b class="mono">{int((map_df["cls"] == c).sum())}</b></div>'
                    for c in sorted(map_df["cls"].unique())
                )
                html(f'<div class="legend">{rows}</div><p class="note">Marker size shows repair priority.</p>')

            shown = map_df[map_df["cls"].isin(classes_on_map)]
            layers = []
            if show_route:
                layers.append(pdk.Layer(
                    "PathLayer", data=[{"path": route_pts}], get_path="path",
                    get_color=hex_to_rgba(T["text"], 220), width_min_pixels=3,
                ))
            if show_heat and len(shown):
                layers.append(pdk.Layer(
                    "HeatmapLayer", data=shown, get_position="[lon, lat]", get_weight="score",
                    radius_pixels=50, opacity=0.55,
                ))
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=shown, get_position="[lon, lat]", get_fill_color="color",
                get_radius="radius", radius_units="pixels", stroked=True, get_line_color=[255, 255, 255, 200],
                line_width_min_pixels=1, pickable=True, auto_highlight=True,
            ))
            view = pdk.ViewState(latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=14.5, pitch=40, bearing=0)
            deck = pdk.Deck(
                layers=layers, initial_view_state=view, map_provider="carto", map_style=T["map"],
                tooltip={
                    "html": "<b>#{id} · {label}</b><br/>Priority: {priority}<br/>Peak confidence: {conf_txt}<br/>Chainage: {km} km",
                    "style": {"backgroundColor": T["surface"], "color": T["text"], "border": f"1px solid {T['border']}",
                              "borderRadius": "10px", "fontFamily": "Inter", "fontSize": "12px"},
                },
            )
            with m_col:
                st.pydeck_chart(deck, use_container_width=True, height=560)
            st.caption("Positions are dead-reckoned from video time, the configured start point, bearing and speed. "
                       "Use the Geo-reference settings to align the route; plug in real GPS logs for survey-grade accuracy.")
        else:
            st.info("No hazards detected on this route.")

    # ---------------- Analytics ----------------
    with tab_analytics:
        if records:
            df = pd.DataFrame(records)
            df["Class"] = df["cls"].str.capitalize()
            classes = sorted(df["Class"].unique())
            class_scale = alt.Scale(domain=classes, range=[class_color(c.lower(), T) for c in classes])

            route_m = max(1.0, results["route_m"])
            seg = 100 if route_m <= 3000 else 250 if route_m <= 10000 else 500
            df["Segment"] = (df["chainage_m"] // seg * seg / 1000).round(2)
            worst = df.groupby("Segment").size().sort_values(ascending=False)

            html(
                '<div class="tiles">'
                f'<div class="tile"><div class="l">Hazards / km</div><div class="v">{results["density"]:.1f}</div></div>'
                f'<div class="tile"><div class="l">High priority</div><div class="v" style="color:var(--pothole)">{int((df["priority"] == "High").sum())}</div></div>'
                f'<div class="tile"><div class="l">Mean confidence</div><div class="v">{df["conf"].mean():.0%}</div></div>'
                f'<div class="tile"><div class="l">Worst {seg} m segment</div><div class="v">km {worst.index[0]:.2f}</div></div>'
                f'<div class="tile"><div class="l">Avg speed</div><div class="v">{results["avg_fps"]:.1f} fps</div></div>'
                "</div>"
            )
            st.write("")

            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                html(f'<div class="panel-title"><h3>Hazards per {seg} m segment</h3><span>Where to send crews first.</span></div>')
                seg_df = df.groupby(["Segment", "Class"]).size().reset_index(name="Count")
                bars = alt.Chart(seg_df).mark_bar().encode(
                    x=alt.X("Segment:O", title="Chainage (km)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("sum(Count):Q", title="Hazards", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("Class:N", scale=class_scale, legend=alt.Legend(title=None)),
                    tooltip=["Segment", "Class", "Count"],
                ).properties(height=300)
                st.altair_chart(style_chart(bars, T), use_container_width=True, theme=None)
            prio_color = alt.Color("priority:N", sort=PRIORITY_ORDER, title=None,
                                   scale=alt.Scale(domain=PRIORITY_ORDER, range=[priority_color(p, T) for p in PRIORITY_ORDER]))
            class_color_enc = alt.Color("Class:N", scale=class_scale, legend=alt.Legend(title=None))

            def show(chart, height=280):
                st.altair_chart(style_chart(chart.properties(height=height), T), use_container_width=True, theme=None)

            with c2:
                html('<div class="panel-title"><h3>Repair priority by hazard type</h3><span>Priority combines confidence and apparent size.</span></div>')
                show(alt.Chart(df).mark_bar().encode(
                    y=alt.Y("Class:N", title=None),
                    x=alt.X("count():Q", title="Hazards", axis=alt.Axis(tickMinStep=1)),
                    color=prio_color,
                    tooltip=["Class", alt.Tooltip("priority:N", title="Priority"), alt.Tooltip("count():Q", title="Hazards")],
                ), 300)

            c3, c4 = st.columns(2, gap="large")
            with c3:
                html('<div class="panel-title"><h3>Detection confidence</h3><span>How sure the model was about each hazard, at its best frame.</span></div>')
                show(alt.Chart(df).mark_bar().encode(
                    x=alt.X("conf:Q", bin=alt.Bin(step=0.1, extent=[0, 1]), title="Peak confidence", axis=alt.Axis(format="%")),
                    y=alt.Y("count():Q", title="Hazards", axis=alt.Axis(tickMinStep=1)),
                    color=class_color_enc, tooltip=["Class", alt.Tooltip("count():Q", title="Hazards")],
                ))
            with c4:
                html('<div class="panel-title"><h3>New hazards per second of video</h3><span>When along the drive hazards first appeared.</span></div>')
                df["Second"] = df["time_s"].astype(int)
                show(alt.Chart(df).mark_bar().encode(
                    x=alt.X("Second:O", title="Video time (s)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("count():Q", title="New hazards", axis=alt.Axis(tickMinStep=1)),
                    color=class_color_enc, tooltip=["Second", "Class", alt.Tooltip("count():Q", title="New hazards")],
                ))

            c5, c6 = st.columns([3, 2], gap="large")
            with c5:
                html('<div class="panel-title"><h3>Hazard profile along the route</h3><span>Bubble size shows apparent size in the frame.</span></div>')
                show(alt.Chart(df).mark_circle(opacity=0.85, stroke=T["surface"], strokeWidth=1).encode(
                    x=alt.X("chainage_m:Q", title="Chainage (m)"),
                    y=alt.Y("conf:Q", title="Peak confidence", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    size=alt.Size("area_pct:Q", scale=alt.Scale(range=[40, 600]), legend=None),
                    color=class_color_enc,
                    tooltip=[alt.Tooltip("id:N", title="ID"), "Class", alt.Tooltip("priority:N", title="Priority"),
                             alt.Tooltip("conf:Q", format=".0%", title="Confidence"),
                             alt.Tooltip("chainage_m:Q", title="Chainage (m)")],
                ), 320)
            with c6:
                html('<div class="panel-title"><h3>Repair shortlist</h3><span>The ten most urgent hazards by priority score.</span></div>')
                top = df.nlargest(10, "score").copy()
                top["Hazard"] = top["Class"] + " " + top["id"].astype(str)
                show(alt.Chart(top).mark_bar().encode(
                    y=alt.Y("Hazard:N", sort="-x", title=None),
                    x=alt.X("score:Q", title="Priority score", scale=alt.Scale(domain=[0, 1])),
                    color=prio_color,
                    tooltip=["Hazard", alt.Tooltip("score:Q", format=".2f", title="Score"),
                             alt.Tooltip("conf:Q", format=".0%", title="Confidence"),
                             alt.Tooltip("chainage_m:Q", title="Chainage (m)")],
                ), 320)

            c7, c8 = st.columns(2, gap="large")
            with c7:
                html('<div class="panel-title"><h3>Tracking persistence</h3><span>Frames each hazard was followed; longer tracks are more reliable.</span></div>')
                show(alt.Chart(df).mark_bar().encode(
                    x=alt.X("frames:Q", bin=alt.Bin(maxbins=12), title="Frames tracked"),
                    y=alt.Y("count():Q", title="Hazards", axis=alt.Axis(tickMinStep=1)),
                    color=class_color_enc, tooltip=["Class", alt.Tooltip("count():Q", title="Hazards")],
                ), 240)
            with c8:
                html('<div class="panel-title"><h3>Hazard size</h3><span>Largest share of the frame each hazard covered.</span></div>')
                show(alt.Chart(df).mark_bar().encode(
                    x=alt.X("area_pct:Q", bin=alt.Bin(maxbins=12), title="Size (% of frame)"),
                    y=alt.Y("count():Q", title="Hazards", axis=alt.Axis(tickMinStep=1)),
                    color=class_color_enc, tooltip=["Class", alt.Tooltip("count():Q", title="Hazards")],
                ), 240)
        else:
            st.info("No hazards to analyse.")

    # ---------------- Registry ----------------
    with tab_registry:
        if records:
            f1, f2, f3 = st.columns([2, 2, 2])
            all_cls = sorted({r["cls"] for r in records})
            sel_cls = f1.multiselect("Class", all_cls, default=all_cls, format_func=str.capitalize)
            sel_pr = f2.multiselect("Priority", PRIORITY_ORDER, default=PRIORITY_ORDER)
            sort_by = f3.selectbox("Sort by", ["Chainage", "Priority score", "Confidence", "Size"])
            key = {"Chainage": ("chainage_m", False), "Priority score": ("score", True),
                   "Confidence": ("conf", True), "Size": ("area_pct", True)}[sort_by]
            view_recs = sorted([r for r in records if r["cls"] in sel_cls and r["priority"] in sel_pr],
                               key=lambda r: r[key[0]], reverse=key[1])

            html(f'<div class="panel-title"><h3>Hazard registry</h3><span>{len(view_recs)} of {len(records)} hazards</span></div>')
            html(registry_table_html(view_recs, T))
            st.write("")

            export_df = pd.DataFrame(view_recs).rename(columns={
                "id": "hazard_id", "cls": "class", "conf": "peak_confidence", "area_pct": "peak_size_pct",
                "score": "priority_score", "time_s": "video_time_s", "frames": "frames_tracked",
            })
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                     "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}}
                    for r in view_recs
                ],
            }
            e1, e2, e3 = st.columns(3)
            e1.download_button("CSV report", export_df.to_csv(index=False).encode("utf-8"),
                               f"road_hazards_{stem}.csv", "text/csv", use_container_width=True)
            e2.download_button("JSON report", json.dumps(view_recs, indent=2, default=str).encode("utf-8"),
                               f"road_hazards_{stem}.json", "application/json", use_container_width=True)
            e3.download_button("GeoJSON for QGIS / ArcGIS", json.dumps(geojson, indent=2, default=str).encode("utf-8"),
                               f"road_hazards_{stem}.geojson", "application/geo+json", use_container_width=True)
        else:
            st.info("No hazards recorded.")

    # ---------------- Evidence ----------------
    with tab_evidence:
        ev = results["evidence"]
        ev_recs = [r for r in records if r["id"] in ev]
        if ev_recs:
            g1, g2 = st.columns([2, 4])
            ev_cls = g1.selectbox("Filter", ["All"] + sorted({r["cls"] for r in ev_recs}), format_func=str.capitalize)
            if ev_cls != "All":
                ev_recs = [r for r in ev_recs if r["cls"] == ev_cls]
            ev_recs = sorted(ev_recs, key=lambda r: r["score"], reverse=True)[:60]
            html(f'<div class="panel-title"><h3>Best-confidence snapshot per hazard</h3><span>ranked by repair priority · {len(ev_recs)} shown</span></div>')
            per_row = 5
            for i in range(0, len(ev_recs), per_row):
                cols = st.columns(per_row)
                for col, r in zip(cols, ev_recs[i : i + per_row]):
                    with col:
                        st.image(ev[r["id"]], use_container_width=True)
                        html(
                            f'<div class="ev-cap"><span><b>{r["cls"].capitalize()} {r["id"]}</b></span>'
                            f'<span class="pill" style="--c:{priority_color(r["priority"], T)}">{r["priority"]}</span></div>'
                        )
        else:
            st.info("No evidence snapshots captured.")