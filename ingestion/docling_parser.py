import logging
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str, force_ocr: bool = False):
    """
    Convert a PDF to a Docling document object.

    Docling handles both digital and scanned PDFs automatically.
    For digital PDFs, OCR is disabled by default (most reports are digital).
    Set force_ocr=True for scanned/image-based PDFs that need OCR.
    Reduced batch sizes keep memory usage within CPU limits.
    Returns a DoclingDocument with .tables, .pictures, and export methods.
    """
    logger.info(f"Parsing PDF: {file_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = force_ocr
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 0.75  # Reduce page image resolution to save memory
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=1)
    pipeline_options.force_backend_text = not force_ocr

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(file_path)
    logger.info(f"Parsing complete. Pages: {len(result.document.pages)}")
    return result.document


def get_markdown(document) -> str:
    """
    Export the full document as clean markdown.
    Tables are rendered as markdown tables. Headings are preserved.
    Used to generate the topic summary.
    """
    return document.export_to_markdown()
