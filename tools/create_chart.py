import os
from pathlib import Path
import threading
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type
from contextlib import redirect_stdout
import traceback
import io
import re
from agent_logger import tool_logger

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Anchor to the project root (the folder above tools/), not the current working directory,
# so charts land in the same place whether this runs from the notebook, main.py or elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHART_ROOT = PROJECT_ROOT / "charts"

def report_relative(path: Path, report_dir: Path = PROJECT_ROOT) -> str:
    """Path for <iframe src>: relative to the folder the HTML report is saved in."""
    return Path(os.path.relpath(Path(path).resolve(), Path(report_dir).resolve())).as_posix()

_CHART_LOCK = threading.Lock()

def _house_layout(fig: go.Figure, title: str) -> None:
    is_line = all(t.type == "scatter" and "lines" in (t.mode or "") for t in fig.data)
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Microsoft YaHei, PingFang SC, Hiragino Sans GB, sans-serif", size=13, color="#0b0b0b"),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18)),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        margin=dict(l=90, r=48, t=96, b=64),
        showlegend=len(fig.data) > 1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="#fcfcfb", font_size=13),
        hovermode="x unified" if is_line else "closest",
        autosize=True, width=None, height=None,  # fill the iframe instead of a fixed 900px
    )
    fig.update_xaxes(showgrid=False, linecolor="#d6d5d0", ticks="outside", tickcolor="#d6d5d0")
    fig.update_yaxes(showgrid=True, gridcolor="#eceae5", zeroline=True, zerolinecolor="#d6d5d0")
    # Labels on the longest bar or last point would otherwise be cut off at the axis edge
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_traces(cliponaxis=False, selector=dict(type="scatter"))
    _pad_bar_value_axis(fig)

def _pad_bar_value_axis(fig: go.Figure) -> None:
    """Widen the value axis so labels outside the bars (esp. negative ones) have room."""
    bars = [t for t in fig.data if t.type == "bar" and t.text is not None and t.textposition in (None, "outside", "auto")]
    if not bars or fig.layout.barmode in ("stack", "relative"):
        return
    horizontal = all(t.orientation == "h" for t in bars)
    axis = fig.layout.xaxis if horizontal else fig.layout.yaxis
    if axis.range is not None or axis.type in ("log", "date", "category"):
        return  # the chart code chose its own range
    try:
        values = [float(v) for t in bars for v in ((t.x if horizontal else t.y) or []) if v is not None]
    except (TypeError, ValueError):
        return
    if not values:
        return
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    span = (high - low) or 1.0
    # Room for a label like "-12.4%": scale with the negative side itself, so one large
    # positive outlier does not push the axis far below the smallest bar
    low -= max(0.18 * -low, 0.06 * span) if low < 0 else 0.0
    high += 0.12 * span if high > 0 else 0.0
    (fig.update_xaxes if horizontal else fig.update_yaxes)(range=[low, high])

class CreateChartInput(BaseModel):
    code: str = Field(description=(
        "Python source that builds ONE plotly chart. `go` (plotly.graph_objects) and PALETTE are "
        "already defined. Assign the figure to `fig` and the title string to `title`. "
        "Do not call fig.show() or fig.write_html(); the tool applies the layout and saves the chart."
    ))
    file_name: str = Field(description="chart_p<first data page>_<lowercase-hyphenated-name>, e.g. 'chart_p3_northbound-flows'.")

@tool_logger
class CreateChart(BaseTool):
    name: str = "create_chart"
    description: str = (
        "Runs plotly code to draw a chart that supports a quantitative claim in the report, "
        "saves it as an HTML file and returns the path to embed in the report with <iframe>."
    )
    args_schema: Type[BaseModel] = CreateChartInput
    # Folder for this report's charts; each run gets its own so concurrent runs do not mix charts
    chart_dir: Path = CHART_ROOT

    def _run(self, code: str, file_name: str) -> str:
        # Models sometimes wrap code in a Markdown fence despite the prompt
        code = re.sub(r"^\s*```(?:python|py)?\s*|\s*```\s*$", "", code)
        slug = re.sub(r"[^a-z0-9_-]+", "-", Path(file_name).stem.lower()).strip("-") or "chart"
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.chart_dir / f"{slug}.html"

        # A fresh namespace per call, so one chart can never pick up another chart's fig or title
        namespace = {"go": go, "PALETTE": list(PALETTE)}
        stdout = io.StringIO()
        with _CHART_LOCK:
            try:
                with redirect_stdout(stdout):
                    exec(compile(code, f"<{slug}>", "exec"), namespace)
            except Exception:
                tb = traceback.format_exc(limit=3)
                return f"Chart failed. Fix the code and retry.\n{tb}{stdout.getvalue()}"

        fig, title = namespace.get("fig"), namespace.get("title")
        if not isinstance(fig, go.Figure) or not fig.data:
            return "Chart failed. The code must assign a go.Figure with at least one trace to `fig`."
        if not isinstance(title, str) or not title.strip():
            return "Chart failed. The code must assign the chart title (metric, unit, period) to `title`."

        _house_layout(fig, title)
        fig.write_html(
            out_path, include_plotlyjs="cdn", full_html=True,
            default_width="100%", default_height="100%",
            config=dict(displayModeBar=False, responsive=True),
        )
        src = report_relative(out_path)
        return (
            f"Chart saved: {src}\n"
            f'Embed it with <iframe src="{src}"></iframe>\n'
            f"{stdout.getvalue()}"
        ).strip()
