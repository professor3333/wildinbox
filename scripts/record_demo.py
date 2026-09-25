"""Record the short demonstration video by driving the real review UI.

    uv run --with playwright python scripts/record_demo.py --out docs/media/demo.webm

Needs a running deployment with the trained release active and an empty
database (see docs/demo.md), and a Playwright Chromium
(`uv run --with playwright playwright install chromium`). The script uploads
the committed sample batch through the UI, reviews it, and exports, with a
caption for each step. Nothing is staged: every screen is the live app.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

CAPTION_CSS = """
position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%);
z-index: 2147483647; max-width: 80%; padding: 10px 18px; border-radius: 10px;
background: rgba(20, 24, 28, 0.88); color: #fff; font: 600 20px/1.35 system-ui, sans-serif;
text-align: center; box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
"""


def caption(page: Page, text: str, hold: float = 2.5) -> None:
    page.evaluate(
        """([text, css]) => {
            let el = document.getElementById('demo-caption');
            if (!el) { el = document.createElement('div'); el.id = 'demo-caption';
                        document.documentElement.appendChild(el); }
            el.style.cssText = css; el.textContent = text;
        }""",
        [text, CAPTION_CSS],
    )
    page.wait_for_timeout(int(hold * 1000))


def goto(page: Page, name: str) -> None:
    page.get_by_test_id("stSidebar").get_by_text(name, exact=True).click()
    page.wait_for_timeout(1500)


def settle(page: Page, ms: int = 1500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="http://localhost:8501")
    ap.add_argument("--sample", type=Path, default=Path("samples/cct-dev"))
    ap.add_argument("--out", type=Path, default=Path("docs/media/demo.webm"))
    ap.add_argument("--screenshots", type=Path, default=Path("docs/media"))
    ap.add_argument(
        "--night", default="2011/04/24", help="a night of the sample with an animal event"
    )
    args = ap.parse_args()
    files = sorted(str(p) for p in (args.sample / "images").iterdir())
    args.screenshots.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=tmp,
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.goto(args.ui)
        settle(page, 3000)
        caption(page, "WildInbox: a memory card of trail-camera photos, reviewed as capture events")

        goto(page, "Upload")
        caption(
            page,
            "Upload the card: 37 photos plus the card's metadata (cameras, times, sequences)",
            2,
        )
        inputs = page.locator("input[type=file]")
        inputs.nth(0).set_input_files(files)
        page.wait_for_timeout(1500)
        inputs.nth(1).set_input_files(str(args.sample / "metadata.json"))
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Upload and process").click()
        caption(page, "The API stores the batch and returns at once; a worker scores it", 1)
        page.get_by_text("capture events ready for review").wait_for(timeout=180_000)
        settle(page, 1000)
        caption(page, "Grouped into capture events: frames of one trigger are reviewed together")
        page.screenshot(path=str(args.screenshots / "demo-upload.png"))

        page.get_by_label("Your name (recorded with reviews)").fill("demo-reviewer")
        page.keyboard.press("Enter")
        page.get_by_role("button", name="Review this batch").click()
        settle(page, 2500)
        caption(
            page,
            "Every event is in the review queue: automation is off, because evaluation "
            "on unseen cameras did not support it",
            4,
        )
        page.screenshot(path=str(args.screenshots / "demo-review.png"))
        page.mouse.wheel(0, 500)
        caption(page, "Each event shows its frames, the suggestion, its confidence, and why", 3)

        first = page.get_by_role("button", name="Accept:").first
        first.scroll_into_view_if_needed()
        caption(page, "Accept a correct suggestion in one click", 2)
        first.click()
        settle(page, 2500)

        page.get_by_text("All frames, predictions, and review history").first.click()
        page.wait_for_timeout(1200)
        caption(page, "Every frame's prediction and the full review history stay available", 3)
        page.get_by_text("All frames, predictions, and review history").first.click()

        combo = page.get_by_test_id("stSelectbox").nth(1)
        combo.scroll_into_view_if_needed()
        caption(page, "Or correct it: pick another species (the suggestion is kept)", 2)
        combo.click()
        page.get_by_role("option", name="coyote").click()
        page.get_by_role("button", name="Save").first.click()
        settle(page, 2500)

        goto(page, "Last night's visitors")
        settle(page, 1000)
        # A segmented date field: set year, month, and day, then commit.
        for part, value in zip(("year", "month", "day"), args.night.split("/"), strict=True):
            page.get_by_role("spinbutton", name=f"{part}, Night starting on").click()
            page.keyboard.type(value)
        page.keyboard.press("Tab")
        settle(page, 2500)
        caption(page, "Last night's visitors: pick a night, see the best frame of each animal", 4)
        page.screenshot(path=str(args.screenshots / "demo-visitors.png"))

        goto(page, "Timeline")
        settle(page, 1500)
        caption(page, "A timeline by day and camera (events are captures, not animal counts)", 3)

        goto(page, "Export")
        settle(page)
        page.get_by_role("button", name="Prepare CSV").click()
        page.get_by_role("button", name="Download CSV").wait_for(timeout=60_000)
        caption(
            page,
            "Export the observation log: reviewed labels, with the release and policy "
            "behind each row",
            3.5,
        )

        goto(page, "Monitoring")
        settle(page, 3000)
        caption(page, "Monitoring: queue, failures, latency, workers, and model behavior", 3.5)
        caption(page, "WildInbox: find the wildlife, skip the empty frames", 2.5)

        video = page.video
        context.close()
        browser.close()
        assert video is not None
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(video.path(), args.out)
    print(f"video -> {args.out}")


if __name__ == "__main__":
    main()
