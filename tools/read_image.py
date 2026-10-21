from pydantic import BaseModel, Field
from typing import Type
from langchain.tools import BaseTool
from langchain_deepseek import ChatDeepSeek
import os
import base64
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
import yaml
from agent_logger import tool_logger

VISION_MODEL = "deepseek-flash"

# Resolve prompt.yaml from the project root, not the working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISION_PROMPT = yaml.safe_load((PROJECT_ROOT / "prompt.yaml").read_text(encoding="utf-8"))["vision_prompt"]

class ReadImageInput(BaseModel):
    image: str = Field(description="Image file name or path exactly as it appears in read_pdf output, e.g. 'test.pdf-0012-06.png'.")
    context: str = Field(default="", description="Optional caption/exhibit title/nearby text from the page, to help identify the image.")
    question: str = Field(default="", description="Optional focused question for a follow-up read of a detail. Empty = full transcription.")

@tool_logger
class ReadImage(BaseTool):
    name: str = "read_image"
    description: str = (
        "Sends an image extracted by read_pdf to a vision model and returns a verbatim transcription "
        "of its title, axes, legend, data labels, table cells and footnotes. Unlabelled values are "
        "reported as '未标注', never estimated."
    )
    args_schema: Type[BaseModel] = ReadImageInput
    vision_model: ChatDeepSeek = Field(default_factory=lambda: ChatDeepSeek(
        model=VISION_MODEL, api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0,
    ))
    cache: dict = Field(default_factory=dict)
    # Same folder as read_pdf's image_dir: it writes the images there, this tool reads them
    image_dir: Path | None = None

    def _run(self, image: str, context: str = "", question: str = "") -> str:
        if self.image_dir is None:
            return "No image folder is active; images from read_pdf were not kept."
        # Only look inside the run's image folder, whatever path the agent passes
        path = self.image_dir / Path(image).name
        if not path.exists():
            return f"Image not found: {Path(image).name}. Use a file name from read_pdf output."
        key = (path.name, question)
        if key in self.cache:
            return self.cache[key]

        if question:
            task = f"只回答这个问题，依据仅限图中明确可见的信息，看不到就直接说看不到：\n{question}"
        else:
            task = "按规定结构完整转录这张图片。"
        if context:
            task += f"\n\n页面文字（仅用于辨认图片主题，不得用来补全图中数值）：\n{context}"
        b64 = base64.b64encode(path.read_bytes()).decode()
        mime = f"image/{path.suffix.lstrip('.').replace('jpg', 'jpeg')}"
        response = self.vision_model.invoke([
            SystemMessage(content=VISION_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": task},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]),
        ])
        result = f"[read_image: {path.name}]\n{response.content}"
        self.cache[key] = result
        return result
