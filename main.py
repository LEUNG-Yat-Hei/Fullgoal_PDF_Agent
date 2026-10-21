"""Extract decision-relevant information from a research PDF into an HTML report with charts.

Usage:
    uv run main.py test2.pdf
"""
import argparse
import difflib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import yaml
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_deepseek import ChatDeepSeek

from tools.create_chart import CHART_ROOT, PROJECT_ROOT, CreateChart, report_relative
from tools.read_image import ReadImage
from tools.read_pdf import ReadPdf

REASONING_MODEL = "deepseek-v4-pro"

@wrap_tool_call
def tool_errors_to_agent(request, handler):
    """Return tool exceptions to the agent as the tool result instead of stopping the run."""
    try:
        return handler(request)
    except Exception as e:
        call = request.tool_call
        return ToolMessage(
            content=(
                f"Tool '{call['name']}' failed with {type(e).__name__}: {e}\n"
                "Fix the arguments and retry, try another approach, or continue without this result "
                "and mention the gap under 数据限制."
            ),
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )


def build_agent(image_dir: Path, chart_dir: Path, model: str = REASONING_MODEL):
    prompts = yaml.safe_load((PROJECT_ROOT / "prompt.yaml").read_text(encoding="utf-8"))
    return create_agent(
        model=ChatDeepSeek(model=model, api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0.1),
        tools=[ReadPdf(image_dir=image_dir), ReadImage(image_dir=image_dir), CreateChart(chart_dir=chart_dir)],
        system_prompt=prompts["system_prompt"] + "\n" + prompts["chart_prompt"],
        middleware=[tool_errors_to_agent],
    )


def run_agent(agent, pdf_path: Path, on_tool_call: Callable[[str], None] | None = None) -> str:
    """Stream the agent run, reporting each tool call to on_tool_call, and return the text of its final message."""
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": f"Read {pdf_path.as_posix()}"}]},
        version="v3",
    )
    final_text = ""
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            # Each new message resets the buffer, so it ends up holding the final answer
            final_text = ""
            for token in item.text:
                # print(token, end="", flush=True)
                final_text += token
        elif kind == "tool_calls" and on_tool_call is not None:
            on_tool_call(item.tool_name)
        # elif kind == "tool_calls":
        #     # print(f"\nTool call: {item.tool_name}({item.input})")
        #     for delta in item.output_deltas:
                # print(delta, end="", flush=True)
            # print(f"\nTool result: {item.output}")
    return final_text


def extract_html(final_text: str) -> str:
    """Keep only the HTML document; the model sometimes adds a sentence or a ``` fence around it."""
    match = re.search(r"<!DOCTYPE html.*</html>", final_text, re.IGNORECASE | re.DOTALL)
    return match.group(0) if match else re.sub(r"^\s*```(?:html)?\s*|\s*```\s*$", "", final_text)


def check_chart_links(html: str, chart_dir: Path) -> str:
    """Repair mistyped chart paths, drop figures whose chart does not exist, and list unused charts."""
    saved = sorted(report_relative(path) for path in chart_dir.glob("*.html"))
    used = set()

    def fix_figure(m: re.Match) -> str:
        src = m.group(1)
        if src in saved:
            used.add(src)
            return m.group(0)
        close = difflib.get_close_matches(src, saved, n=1, cutoff=0.8)
        if close:
            print(f"Fixed chart link: {src} -> {close[0]}")
            used.add(close[0])
            return m.group(0).replace(src, close[0])
        print(f"WARNING: removed figure, chart file not found: {src}")
        return ""

    figure = r'<figure>(?:(?!</figure>).)*?<iframe[^>]*?src="([^"]+)"(?:(?!</figure>).)*?</figure>'
    html = re.sub(figure, fix_figure, html, flags=re.DOTALL)
    for src in sorted(set(saved) - used):
        print(f"WARNING: chart was created but not embedded in the report: {src}")
    return html


CHART_CSS = ".chart { width: 100%; max-width: 920px; height: 540px; }"


def inline_charts(html: str) -> str:
    """Replace each chart <iframe> with the chart's own <div>+<script>, loading plotly.js once in <head>."""
    plotly_scripts = set()

    def inline(m: re.Match) -> str:
        chart_file = PROJECT_ROOT / m.group(1)
        body = re.search(r"<body>(.*)</body>", chart_file.read_text(encoding="utf-8"), re.DOTALL).group(1)
        # The plotly.js <script src> goes to <head> once instead of once per chart
        script_tag = r'<script[^>]*\bsrc="[^"]*"[^>]*>\s*</script>'
        plotly_scripts.update(re.findall(script_tag, body))
        return f'<div class="chart">{re.sub(script_tag, "", body).strip()}</div>'

    html = re.sub(r'<iframe[^>]*?src="([^"]+)"[^>]*>\s*</iframe>', inline, html)
    if plotly_scripts:
        head = f"<style>{CHART_CSS}</style>\n" + "\n".join(sorted(plotly_scripts)) + "\n</head>"
        html = re.sub(r"</head>", lambda _: head, html, count=1, flags=re.IGNORECASE)
    return html


def generate_report(
    pdf_path: Path, model: str = REASONING_MODEL, on_tool_call: Callable[[str], None] | None = None
) -> str:
    """Run the agent on one PDF and return the finished HTML report with its charts inlined."""
    # Each run gets its own image and chart folders, so concurrent runs (e.g. two web users) never mix files.
    # Charts stay under CHART_ROOT because the agent embeds them by paths relative to the project root.
    CHART_ROOT.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory() as image_dir,
        tempfile.TemporaryDirectory(dir=CHART_ROOT, prefix=f"{pdf_path.stem}-") as chart_dir,
    ):
        chart_dir = Path(chart_dir)
        agent = build_agent(Path(image_dir), chart_dir, model)
        final_text = run_agent(agent, pdf_path, on_tool_call)
        # Inline before the folder is removed: the charts then live inside the report itself
        return inline_charts(check_chart_links(extract_html(final_text), chart_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path, help="path to the PDF to summarise")
    parser.add_argument("--model", default=REASONING_MODEL, help=f"reasoning model (default: {REASONING_MODEL})")
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        parser.error(f"PDF not found: {pdf_path}")

    load_dotenv(PROJECT_ROOT / ".env")
    html = generate_report(pdf_path, args.model)
    output_path = PROJECT_ROOT / f"{pdf_path.stem}.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
