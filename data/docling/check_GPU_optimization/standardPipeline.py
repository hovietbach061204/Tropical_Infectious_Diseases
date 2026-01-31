import logging
import logging
import time

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

_log = logging.getLogger(__name__)

def main():
    logging.getLogger("docling").setLevel(logging.WARNING)
    _log.setLevel(logging.INFO)

    input_doc_path = "../Tropical_Dataset/case96.pdf"

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.MPS,
        ),
        ocr_batch_size=4,
        layout_batch_size=64,
        table_batch_size=4,
    )
    pipeline_options.do_ocr = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # start_time = time.time()
    # doc_converter.initialize_pipeline(InputFormat.PDF)
    # init_runtime = time.time() - start_time
    # _log.info(f"Pipeline initialized in {init_runtime:.2f} seconds.")

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    pipeline_runtime = time.time() - start_time
    assert conv_result.status == ConversionStatus.SUCCESS

    num_pages = len(conv_result.pages)
    _log.info(f"Document converted in {pipeline_runtime:.2f} seconds.")
    _log.info(f"  {num_pages / pipeline_runtime:.2f} pages/second.")

    result = conv_result.document.export_to_markdown()
    output_path = "../output/output_simple_case96_v4.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"\n✓ Full markdown saved to: {output_path}")
    print(f"✓ Total length: {len(result)} characters")


if __name__ == "__main__":
    main()