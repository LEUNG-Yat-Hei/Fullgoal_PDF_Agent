"""Web frontend: upload a research PDF and get the HTML report with charts.

Usage:
    uv run streamlit run app.py
"""
import html
import json
import re
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from main import PROJECT_ROOT, generate_report

load_dotenv(PROJECT_ROOT / ".env")

TOOL_LABELS = {
    "read_pdf": "Reading PDF pages",
    "read_image": "Reading a chart or table image",
    "create_chart": "Drawing a chart",
}

# Injected into the previewed report only (the download is untouched). The report is written for a white page,
# so in a dark app its text is recoloured; the wrapper frame below posts the app's current theme into it.
PREVIEW_THEME_HOOK = """
<style>
html[data-theme="dark"] { color-scheme: dark; }
html[data-theme="dark"] body, html[data-theme="dark"] h1, html[data-theme="dark"] h2, html[data-theme="dark"] h3,
html[data-theme="dark"] h4, html[data-theme="dark"] p, html[data-theme="dark"] li, html[data-theme="dark"] td {
  color: #e6e6e6 !important; background-color: transparent !important; }
html[data-theme="dark"] th { color: #e6e6e6 !important; background-color: #262730 !important; }
html[data-theme="dark"] * { border-color: #3d4050 !important; }
html[data-theme="dark"] a, html[data-theme="dark"] summary { color: #6cb4ff !important; }
html[data-theme="dark"] .page, html[data-theme="dark"] .none, html[data-theme="dark"] figcaption { color: #9aa0a6 !important; }
html[data-theme="dark"] .chart { border-radius: 6px; overflow: hidden; }
</style>
<script>
addEventListener("message", (e) => {
  if (e.data && e.data.reportTheme) document.documentElement.dataset.theme = e.data.reportTheme;
});
</script>
"""

# Runs in the st.iframe wrapper, which is same-origin with the app, so it can read the app's background colour.
# Polling picks up theme switches from the settings menu without a rerun.
PREVIEW_THEME_SYNC = """
<script>
const report = document.getElementById("report");
function appTheme() {
  try {
    const doc = window.parent.document;
    for (const el of [doc.querySelector(".stApp"), doc.querySelector(".stAppViewContainer"), doc.body]) {
      if (!el) continue;
      const [r, g, b, a = 1] = (getComputedStyle(el).backgroundColor.match(/[\\d.]+/g) || []).map(Number);
      if (r !== undefined && a > 0) return 0.299 * r + 0.587 * g + 0.114 * b < 128 ? "dark" : "light";
    }
  } catch (e) {}
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function sync() { report.contentWindow?.postMessage({ reportTheme: appTheme() }, "*"); }
report.addEventListener("load", sync);
setInterval(sync, 1000);
</script>
"""

# Runs in a hidden same-origin frame. Clicking "Generate report" asks for notification permission (browsers want
# a user gesture for that); NOTICE, when set, is shown as a desktop notification once the run has finished.
BROWSER_HOOKS = """
<script>
const app = window.parent;
window.frameElement?.closest('[data-testid="stElementContainer"]')?.style.setProperty("display", "none");

if (app.__reportPermissionHook) app.document.removeEventListener("click", app.__reportPermissionHook, true);
app.__reportPermissionHook = (e) => {
  if ("Notification" in app && app.Notification.permission === "default" && e.target.closest?.(".st-key-generate button")) {
    app.Notification.requestPermission();
  }
};
app.document.addEventListener("click", app.__reportPermissionHook, true);

const notice = NOTICE;
if (notice) {
  if ("Notification" in app && app.Notification.permission === "granted") {
    const n = new app.Notification(notice.title, { body: notice.body, tag: "research-report" });
    n.onclick = () => { app.focus(); n.close(); };
  }
  if (app.document.hidden) {
    const title = app.document.title;
    app.document.title = "✅ " + notice.title;
    app.document.addEventListener("visibilitychange", () => { app.document.title = title; }, { once: true });
  }
}
</script>
"""


def safe_file_name(name: str) -> str:
    """Keep the uploaded file's name readable but free of path parts and odd characters."""
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem).strip("._") or "report"
    return f"{stem}.pdf"


def sandboxed_preview(report: str, height: int = 900) -> None:
    """Show the report in a sandboxed iframe: its scripts (plotly) run, but it cannot reach the Streamlit page.

    The report is LLM output built from an uploaded PDF, so it is treated as untrusted HTML.
    """
    themed = re.sub(r"</head>", lambda _: PREVIEW_THEME_HOOK + "</head>", report, count=1, flags=re.IGNORECASE)
    if themed == report:
        themed = PREVIEW_THEME_HOOK + report
    wrapper = (
        "<style>html, body { margin: 0; background: transparent; }</style>"
        '<iframe id="report" sandbox="allow-scripts allow-popups" '
        f'style="width:100%;height:{height - 10}px;border:0" srcdoc="{html.escape(themed, quote=True)}"></iframe>'
        + PREVIEW_THEME_SYNC
    )
    st.iframe(wrapper, height=height)


def browser_hooks(notice: dict | None) -> None:
    """Install the notification-permission hook and, after a run, notify the user's device."""
    # "<\/" keeps an error message containing "</script>" from closing the script tag early
    st.iframe(BROWSER_HOOKS.replace("NOTICE", json.dumps(notice).replace("</", "<\\/")), height=1)


def start_generation() -> None:
    """Button callback: runs before the rerun, so the widgets below already render disabled."""
    pdf = st.session_state.pdf
    st.session_state.job = {"name": pdf.name, "data": pdf.getvalue()}
    st.session_state.generating = True
    st.session_state.pop("report", None)
    st.session_state.pop("last_run", None)


st.set_page_config(page_title="Research Report Agent", page_icon="📊", layout="wide")
generating = st.session_state.setdefault("generating", False)
browser_hooks(st.session_state.pop("notice", None))

st.title("Research Report Agent")
st.caption("Upload a research PDF and get an HTML report of the decision-relevant points, with charts.")

uploaded = st.file_uploader("Research PDF", type="pdf", key="pdf", disabled=generating)
st.button(
    "Generating report…" if generating else "Generate report",
    key="generate", type="primary", disabled=generating or uploaded is None, on_click=start_generation,
)

if generating:
    job = st.session_state.job
    pdf_name = safe_file_name(job["name"])
    started = time.perf_counter()
    try:
        with st.status("Generating report… this usually takes a few minutes. You'll get a notification when it's done.",
                       expanded=True) as status:
            counts: dict[str, int] = {}
            log: list[str] = []

            def on_tool_call(tool_name: str) -> None:
                counts[tool_name] = counts.get(tool_name, 0) + 1
                line = f"`{time.perf_counter() - started:5.0f}s` {TOOL_LABELS.get(tool_name, tool_name)}"
                log.append(line)
                st.write(line)

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    pdf_path = Path(tmpdir) / pdf_name
                    pdf_path.write_bytes(job["data"])
                    report = generate_report(pdf_path, on_tool_call=on_tool_call)
            except Exception as e:
                st.session_state.last_run = {"label": "Report generation failed", "state": "error", "log": log, "error": e}
                st.session_state.notice = {"title": "Report generation failed", "body": f"{job['name']}: {e}"}
            else:
                elapsed = time.perf_counter() - started
                charts = counts.get("create_chart", 0)
                label = f"Report ready in {elapsed:.0f}s ({charts} chart{'s' if charts != 1 else ''} drawn)"
                st.session_state.last_run = {"label": label, "state": "complete", "log": log}
                st.session_state.report = {"html": report, "name": f"{Path(pdf_name).stem}.html"}
                st.session_state.notice = {"title": "Research report ready", "body": f"{job['name']} — {label}"}
    finally:
        # Also reached if the run is stopped, so the widgets never stay disabled
        st.session_state.generating = False
        st.session_state.pop("job", None)
    # Rerun so the uploader and button render enabled again and the notification fires
    st.rerun()

# The finished run's progress log, kept in session_state because the rerun above clears the live status box
if last_run := st.session_state.get("last_run"):
    with st.status(last_run["label"], state=last_run["state"], expanded=last_run["state"] == "error"):
        for line in last_run["log"]:
            st.write(line)
        if "error" in last_run:
            st.exception(last_run["error"])

# Kept in session_state so the report survives reruns, e.g. when the download button is clicked
if "report" in st.session_state:
    report = st.session_state.report
    st.download_button("Download HTML report", report["html"], file_name=report["name"], mime="text/html")
    sandboxed_preview(report["html"])
