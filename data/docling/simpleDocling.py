"""
Simple PDF Parsing with Docling - Quick Start
==============================================

This script demonstrates the most basic usage of Docling:
converting a PDF to Markdown.

Why Docling?
- Handles complex PDFs with tables, images, and multi-column layouts
- No need for custom OCR implementation
- Preserves document structure and formatting
- Works out-of-the-box without configuration
"""
import logging
import time

from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)
def main():
    logging.getLogger("docling").setLevel(logging.WARNING)
    _log.setLevel(logging.INFO)
    # Path to PDF document
    pdf_path = "../Tropical_Dataset/case101.pdf"

    print("=" * 60)
    print("Converting PDF to Markdown with Docling")
    print("=" * 60)
    print(f"Input: {pdf_path}\n")

    pipeline_options = PdfPipelineOptions(do_table_structure=True, enable_remote_services=True)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    # pipeline_options.table_structure_options.do_cell_matching = False  # uses text cells predicted from table structure model
    pipeline_options.do_picture_description = False  # enable enrichment
    # pipeline_options.picture_description_options = smolvlm_picture_description  # local VLM
    pipeline_options.generate_picture_images = False  # export picture crops
    # pipeline_options.images_scale = 2.0  # better crops (optional)
    # Initialize converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert PDF
    print("Processing PDF...")
    result = converter.convert(pdf_path)

    # Export to Markdown
    markdown = result.document.export_to_markdown()

    # Display results
    print("\n" + "=" * 60)
    print("MARKDOWN OUTPUT")
    print("=" * 60)
    print(markdown[:1000])  # Show first 1000 characters
    print("\n... (truncated for display)")

    # Save to file
    output_path = "output/output_case101.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    start_time = time.time()
    converter.initialize_pipeline(InputFormat.PDF)
    init_runtime = time.time() - start_time
    _log.info(f"Pipeline initialized in {init_runtime:.2f} seconds.")

    start_time = time.time()
    conv_result = converter.convert(pdf_path)
    pipeline_runtime = time.time() - start_time
    assert conv_result.status == ConversionStatus.SUCCESS

    num_pages = len(conv_result.pages)
    _log.info(f"Document converted in {pipeline_runtime:.2f} seconds.")
    _log.info(f"  {num_pages / pipeline_runtime:.2f} pages/second.")

    print(f"\n✓ Full markdown saved to: {output_path}")
    print(f"✓ Total length: {len(markdown)} characters")

if __name__ == "__main__":
    main()