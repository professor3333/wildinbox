"""Minimal server-rendered pages: an upload form and a batch-status view.

Every user-supplied value (filenames, camera ids, errors) is HTML-escaped.
"""

from __future__ import annotations

from html import escape
from typing import Any

from wildinbox.settings import Settings

_STYLE = """
:root { --bg:#fbfaf7; --fg:#1d2320; --muted:#5d6661; --line:#dcd8cf; --warn-bg:#fff4d6;
  --warn-fg:#6b4a00; --bad:#a3261b; --ok:#1f6b3a; --accent:#2f5d50; }
@media (prefers-color-scheme: dark) { :root { --bg:#141816; --fg:#e7ebe8; --muted:#9aa39e;
  --line:#2c332f; --warn-bg:#3a2f12; --warn-fg:#f3d58a; --bad:#ff8a7d; --ok:#7fd49c;
  --accent:#8fc7b5; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 28px 0 8px; }
.muted { color: var(--muted); } a { color: var(--accent); }
.banner { background:var(--warn-bg); color:var(--warn-fg); padding:10px 14px;
  border-radius:8px; margin:16px 0; font-weight:600; }
.stats { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
.stat { border:1px solid var(--line); border-radius:8px; padding:8px 12px; min-width:92px; }
.stat b { display:block; font-size:20px; }
.table-wrap { overflow-x:auto; }
table { border-collapse: collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
  vertical-align: top; }
.bad { color: var(--bad); } .ok { color: var(--ok); }
code { font-size: 13px; }
form { border:1px solid var(--line); border-radius:8px; padding:16px; display:grid; gap:12px; }
input, button { font: inherit; } button { padding:8px 14px; border-radius:6px;
  border:1px solid var(--accent); background:var(--accent); color:var(--bg); cursor:pointer; }
"""


def _page(title: str, body: str, refresh: bool = False) -> str:
    meta = '<meta http-equiv="refresh" content="3">' if refresh else ""
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">{meta}'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )


def upload_page(settings: Settings) -> str:
    limit_mb = settings.max_file_bytes // (1024 * 1024)
    return _page(
        "WildInbox upload",
        f"""
<h1>WildInbox</h1>
<p class="muted">Find the wildlife. Skip the empty frames.</p>
<h2>Upload a batch</h2>
<form id="upload">
  <label>Photos (JPEG or PNG, up to {settings.max_files_per_batch} files,
    {limit_mb} MiB each)<br><input type="file" name="files" multiple required></label>
  <label>Camera id (optional)<br><input type="text" name="camera" maxlength="200"></label>
  <div><button type="submit">Upload</button> <span id="msg" class="muted"></span></div>
</form>
<script>
document.getElementById("upload").addEventListener("submit", async (ev) => {{
  ev.preventDefault();
  const form = ev.target, msg = document.getElementById("msg");
  const body = new FormData();
  for (const f of form.files.files) body.append("files", f);
  if (form.camera.value) body.append("metadata", JSON.stringify({{camera_id: form.camera.value}}));
  msg.textContent = "Uploading...";
  const res = await fetch("/batches", {{method: "POST", body}});
  const data = await res.json();
  if (res.ok) window.location = data.links.view;
  else msg.textContent = data.detail || "Upload failed";
}});
</script>""",
    )


def _stat(label: str, value: Any) -> str:
    return (
        f'<div class="stat"><span class="muted">{escape(label)}</span>'
        f"<b>{escape(str(value))}</b></div>"
    )


def batch_page(
    summary: dict[str, Any], images: list[dict[str, Any]], events: list[dict[str, Any]]
) -> str:
    c, job, release = summary["counts"], summary["job"] or {}, summary["release"] or {}
    running = summary["status"] in ("queued", "processing")
    status_cls = "bad" if summary["status"] == "failed" else ""
    parts = [
        '<p><a href="/">Upload another batch</a></p>',
        f"<h1>Batch <code>{escape(summary['id'])}</code></h1>",
        f'<p>Status: <b class="{status_cls}">{escape(summary["status"])}</b>'
        f"{' (refreshing)' if running else ''} · created {escape(summary['created_at'])}</p>",
    ]
    if release.get("notice"):
        parts.append(f'<div class="banner">{escape(release["notice"])}</div>')
    if summary.get("error"):
        parts.append(f'<p class="bad">Error: {escape(summary["error"])}</p>')
    parts.append(
        '<div class="stats">'
        + "".join(
            [
                _stat("Images", c["images"]),
                _stat("Valid", c["valid"]),
                _stat("Pending", c["pending"]),
                _stat("Invalid", c["invalid"]),
                _stat("Duplicate", c["duplicate"]),
                _stat("Events", c["events"]),
            ]
        )
        + "</div>"
    )
    parts.append(
        f'<p class="muted">Job {escape(str(job.get("id", "")))}: {escape(str(job.get("status")))},'
        f" attempt {escape(str(job.get('attempts')))} of {escape(str(job.get('max_attempts')))}"
        f" · release <code>{escape(str(release.get('id')))}</code>"
        f" · policy <code>{escape(str(release.get('policy_version')))}</code></p>"
    )

    rows = []
    for e in events:
        d = e["decision"] or {}
        conf = d.get("confidence")
        rows.append(
            f"<tr><td>{escape(e['start_at'] or 'no time')}</td>"
            f"<td>{escape(e['camera_id'] or '-')}</td><td>{len(e['image_ids'])}</td>"
            f"<td>{escape(str(d.get('disposition', '-')))}</td>"
            f"<td>{escape(str(d.get('suggested_label', '-')))}"
            f"{'' if conf is None else f' ({conf:.2f})'}</td>"
            f'<td><a href="/events/{escape(e["id"])}">details</a></td></tr>'
        )
    label_hdr = "Suggested label (TEST)" if release.get("is_test") else "Suggested label"
    parts.append("<h2>Capture events</h2>")
    parts.append(
        '<div class="table-wrap"><table><tr><th>Start</th><th>Camera</th><th>Frames</th>'
        f"<th>Disposition</th><th>{label_hdr}</th><th></th></tr>"
        + ("".join(rows) or '<tr><td colspan="6" class="muted">No events yet.</td></tr>')
        + "</table></div>"
    )

    bad = [i for i in images if i["validation_status"] in ("invalid", "duplicate")]
    parts.append("<h2>Files not processed</h2>")
    if bad:
        parts.append(
            '<div class="table-wrap"><table><tr><th>#</th><th>File</th><th>Status</th>'
            "<th>Reason</th></tr>"
            + "".join(
                f"<tr><td>{i['position']}</td><td>{escape(i['filename'])}</td>"
                f'<td class="bad">{escape(i["validation_status"])}</td>'
                f"<td>{escape(i['validation_error'] or ('duplicate of ' + str(i['duplicate_of'])))}"
                "</td></tr>"
                for i in bad
            )
            + "</table></div>"
        )
    else:
        parts.append('<p class="ok">Every file was accepted.</p>')
    return _page(f"Batch {summary['id'][:8]}", "".join(parts), refresh=running)
