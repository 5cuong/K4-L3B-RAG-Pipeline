"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}
MINIMUM_DOCUMENTS = 3


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Reuse the checked-in corpus and fail clearly if it is incomplete.

    The repository already contains the collected legal documents, so this
    task is intentionally offline and does not download duplicate copies.
    Add public-source URLs here if the corpus is replaced with a new topic.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = sorted(
        (
            path
            for path in DATA_DIR.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.stat().st_size > 1024
        ),
        key=lambda path: path.name.lower(),
    )
    if len(documents) < MINIMUM_DOCUMENTS:
        raise RuntimeError(
            f"Found {len(documents)} valid legal documents in {DATA_DIR}; "
            f"at least {MINIMUM_DOCUMENTS} are required. Add public PDF/DOC/DOCX "
            "files before running the collection pipeline."
        )

    print(f"Reusing {len(documents)} checked-in legal documents from {DATA_DIR}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
