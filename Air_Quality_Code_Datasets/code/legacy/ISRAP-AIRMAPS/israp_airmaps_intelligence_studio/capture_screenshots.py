"""
Automated screenshot capture for USER_MANUAL.html.

Requires playwright:
    pip install playwright
    playwright install chromium

Usage (while the app is running on port 8501):
    python capture_screenshots.py

Saves screenshots to: screenshots/
Then patches them into USER_MANUAL.html automatically.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
MANUAL_PATH     = Path(__file__).parent / "USER_MANUAL.html"
APP_URL         = "http://localhost:8501"

PAGES = [
    ("01_executive_overview",  f"{APP_URL}/Executive_Overview",   "Executive Overview"),
    ("02_data_explorer",       f"{APP_URL}/Data_Explorer",        "Data Explorer"),
    ("03_data_quality",        f"{APP_URL}/Data_Quality",         "Data Quality"),
    ("04_anomaly_lab",         f"{APP_URL}/Anomaly_Lab",          "Anomaly Lab"),
    ("05_anomaly_comparison",  f"{APP_URL}/Anomaly_Comparison",   "Anomaly Comparison"),
    ("06_imputation_lab",      f"{APP_URL}/Imputation_Lab",       "Imputation Lab"),
    ("07_imputation_benchmark",f"{APP_URL}/Imputation_Benchmark", "Imputation Benchmark"),
    ("08_final_pipeline",      f"{APP_URL}/Final_Pipeline",       "Final Pipeline"),
    ("09_station3_so2",        f"{APP_URL}/Station3_SO2",         "Station 3 SO₂"),
    ("10_method_library",      f"{APP_URL}/Method_Library",       "Method Library"),
    ("11_experiment_history",  f"{APP_URL}/Experiment_History",   "Experiment History"),
    ("12_export_center",       f"{APP_URL}/Export_Center",        "Export Center"),
    ("13_settings",            f"{APP_URL}/Settings",             "Settings"),
]

VIEWPORT = {"width": 1440, "height": 900}
WAIT_SECONDS = 4   # time for Streamlit page to fully render


def capture(headless: bool = True):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.")
        print("Install with:  pip install playwright && playwright install chromium")
        sys.exit(1)

    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page    = browser.new_page(viewport=VIEWPORT)

        # Load home page first to let Streamlit initialise
        print(f"Connecting to {APP_URL} …")
        page.goto(APP_URL, wait_until="networkidle")
        time.sleep(WAIT_SECONDS)

        for slug, url, label in PAGES:
            print(f"  Capturing: {label} …")
            page.goto(url, wait_until="networkidle")
            time.sleep(WAIT_SECONDS)
            out = SCREENSHOTS_DIR / f"{slug}.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"    saved → {out}")

        browser.close()

    print(f"\nAll {len(PAGES)} screenshots saved to: {SCREENSHOTS_DIR}")
    patch_manual()


def patch_manual():
    """Replace screenshot placeholder divs in USER_MANUAL.html with real <img> tags."""
    if not MANUAL_PATH.exists():
        print(f"WARNING: {MANUAL_PATH} not found — skipping patch.")
        return

    html = MANUAL_PATH.read_text(encoding="utf-8")
    patched = 0

    for slug, _url, label in PAGES:
        img_path = SCREENSHOTS_DIR / f"{slug}.png"
        if not img_path.exists():
            continue

        # Relative path from manual to screenshot
        rel = img_path.relative_to(MANUAL_PATH.parent).as_posix()

        # Replace the first placeholder div for this page
        pattern = (
            r'(<div class="screenshot-placeholder"[^>]*data-page="'
            + re.escape(slug)
            + r'"[^>]*>)(.*?)(</div>)'
        )
        replacement = (
            f'<div class="screenshot-container">'
            f'<img src="{rel}" alt="{label} screenshot" '
            f'style="width:100%;border-radius:8px;border:1px solid #dee2e6;box-shadow:0 4px 12px rgba(0,0,0,.12);">'
            f'<p style="text-align:center;font-size:12px;color:#666;margin-top:6px;">{label}</p>'
            f'</div>'
        )
        new_html, n = re.subn(pattern, replacement, html, flags=re.DOTALL)
        if n:
            html = new_html
            patched += 1

    MANUAL_PATH.write_text(html, encoding="utf-8")
    print(f"Patched {patched} screenshot placeholders in USER_MANUAL.html")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-headless", action="store_true",
                    help="Show browser window while capturing")
    args = ap.parse_args()
    capture(headless=not args.no_headless)

