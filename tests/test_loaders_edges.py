import fitz

from rag_demo.loaders import pdf_loader


def _make_pdf(path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Chapter 1\nThis document explains retrieval augmented generation.",
    )
    document.save(path)
    document.close()


def test_pdf_loader_falls_back_to_pymupdf(tmp_path, monkeypatch):
    path = tmp_path / "fallback.pdf"
    _make_pdf(path)

    class BrokenPyPDFLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            raise RuntimeError("pypdf failed")

    monkeypatch.setattr(pdf_loader, "PyPDFLoader", BrokenPyPDFLoader)

    chunks = pdf_loader.PDFLoader(str(path), chunk_size=100, chunk_overlap=10).load()

    assert chunks
    assert all(chunk.metadata["file_type"] == "pdf" for chunk in chunks)
