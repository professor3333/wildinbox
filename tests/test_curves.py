from __future__ import annotations

import xml.etree.ElementTree as ET

from wildinbox.evaluation.curves import Point, Series, render


def test_curve_is_valid_svg_with_legend_target_and_tooltips() -> None:
    svg = render(
        "Title",
        "coverage",
        "error",
        [
            Series(
                "chosen cameras",
                [Point(0.1, 0.01, 0.0, 0.02, "0.9"), Point(0.3, 0.05, 0.03, 0.08, "0.5")],
                band=True,
            ),
            Series("check cameras", [Point(0.2, 0.04, label="0.7")]),
        ],
        0.02,
        "target 2%",
        Point(0.1, 0.01, label="0.9"),
        "rule: 0.9",
    )
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}polyline")) == 2
    assert len(root.findall(f"{ns}polygon")) == 1  # one interval band
    texts = [t.text for t in root.iter(f"{ns}text")]
    assert texts.count("chosen cameras") == 2  # legend + direct label
    assert "target 2%" in texts and "rule: 0.9" in texts
    assert "prefers-color-scheme: dark" in svg
