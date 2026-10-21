from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Type
from pathlib import Path
import pymupdf
import pymupdf4llm
import tempfile
from agent_logger import tool_logger

IMAGE_DPI = 220

class ReadPdfInput(BaseModel):
    pdf_file: str = Field(description="The file path to the PDF document.")
    pages: List[int] = Field(default_factory=list, description="List of 0-indexed page numbers to read. Empty means all pages.")

@tool_logger
class ReadPdf(BaseTool):
    name: str = "read_pdf"
    description: str = (
        "Converts specific pages of a PDF file into Markdown text. Each page starts with an "
        "'=== Page N of M ===' marker (N is 1-indexed; cite it as p. N). Images appear as ![](...) "
        "links; pass their file names to read_image to read them."
    )
    args_schema: Type[BaseModel] = ReadPdfInput
    # Folder shared with read_image for one agent run: read_pdf writes the images, read_image reads them
    image_dir: Path | None = None

    def _run(self, pdf_file: str, pages: List[int] | None = None) -> str:
        pdf_path = Path(pdf_file)
        with pymupdf.open(pdf_path) as doc:
            page_count = doc.page_count
        kwargs = dict(pages=pages or None, write_images=True, dpi=IMAGE_DPI, page_chunks=True)
        if self.image_dir is not None:
            chunks = pymupdf4llm.to_markdown(pdf_path, image_path=str(self.image_dir), **kwargs)
        else:
            # No run-level folder: images are discarded right away, so read_image cannot use them
            with tempfile.TemporaryDirectory() as tmpdir:
                chunks = pymupdf4llm.to_markdown(pdf_path, image_path=tmpdir, **kwargs)
        return "\n\n".join(
            f"=== Page {chunk['metadata']['page_number']} of {page_count} ===\n{chunk['text']}"
            for chunk in chunks
        )

