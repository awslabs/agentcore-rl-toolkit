"""Strands @tool wrappers for OfficeBench application actions.

Each tool calls the corresponding OfficeBench app script via subprocess.
The app scripts are expected at /apps/ in the Docker container.
All file paths should be relative to or under /testbed/.
"""

import os
import subprocess

from strands import tool

APPS_DIR = os.environ.get("OFFICEBENCH_APPS_DIR", "/apps")
TESTBED_DIR = os.environ.get("OFFICEBENCH_TESTBED_DIR", "/testbed")


def _run_app(app_module: str, script_name: str, args: list[str]) -> str:
    """Run an OfficeBench app script and return its output."""
    cmd = ["python3", f"{APPS_DIR}/{app_module}/{script_name}.py", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TESTBED_DIR, timeout=120)
    output = result.stdout.strip()
    if result.returncode != 0 and result.stderr:
        output = output + "\n" + result.stderr.strip() if output else result.stderr.strip()
    return output or "No output."


# ── Calendar Tools ──────────────────────────────────────────────────────────


@tool
def calendar_create_event(user: str, summary: str, time_start: str, time_end: str) -> str:
    """Create a new calendar event for a user.

    Args:
        user: The username (e.g. 'Bob').
        summary: Event title/summary.
        time_start: Start time in format '%Y-%m-%d %H:%M:%S'.
        time_end: End time in format '%Y-%m-%d %H:%M:%S'.
    """
    return _run_app("calendar_app", "calendar_create_event", [
        "--user", user, "--summary", summary,
        "--time_start", time_start, "--time_end", time_end,
    ])


@tool
def calendar_delete_event(user: str, summary: str) -> str:
    """Delete a calendar event by its summary.

    Args:
        user: The username.
        summary: The event summary to delete.
    """
    return _run_app("calendar_app", "calendar_delete_event", [
        "--user", user, "--summary", summary,
    ])


@tool
def calendar_list_events(username: str) -> str:
    """List all calendar events for a user.

    Args:
        username: The username to list events for.
    """
    return _run_app("calendar_app", "calendar_list_events", ["--username", username])


# ── Email Tools ─────────────────────────────────────────────────────────────


@tool
def email_send_email(sender: str, recipient: str, subject: str, content: str) -> str:
    """Send an email from one user to another.

    Args:
        sender: Sender username (e.g. 'Alice').
        recipient: Recipient username (e.g. 'Bob').
        subject: Email subject line.
        content: Email body content.
    """
    return _run_app("email_app", "email_send_email", [
        "--sender", sender, "--recipient", recipient,
        "--subject", subject, "--content", content,
    ])


@tool
def email_list_emails(username: str) -> str:
    """List all emails for a user. Shows sender, recipient, subject, and a preview.

    Args:
        username: The username to list emails for.
    """
    return _run_app("email_app", "email_list_emails", ["--username", username])


@tool
def email_read_email(username: str, email_id: str) -> str:
    """Read a specific email by its ID (filename).

    Args:
        username: The username whose mailbox to read from.
        email_id: The email filename (e.g. 'Meeting Notes.eml').
    """
    return _run_app("email_app", "email_read_email", [
        "--username", username, "--email_id", email_id,
    ])


# ── Excel Tools ─────────────────────────────────────────────────────────────


@tool
def excel_read_file(file_path: str, sheet: str = "") -> str:
    """Read the contents of an Excel file. Returns cell values in '(row, col): value' format.

    Args:
        file_path: Path to the Excel file (e.g. '/testbed/data/scores.xlsx').
        sheet: Optional sheet name. If empty, reads the active sheet.
    """
    args = ["--file_path", file_path]
    if sheet:
        args.extend(["--sheet", sheet])
    return _run_app("excel_app", "excel_read_file", args)


@tool
def excel_set_cell(file_path: str, text: str, row_idx: int, column_idx: int, sheet_name: str = "") -> str:
    """Write a value to a specific cell in an Excel file. Row and column indices start from 1.

    Args:
        file_path: Path to the Excel file.
        text: The value to write to the cell.
        row_idx: Row index (1-based).
        column_idx: Column index (1-based).
        sheet_name: Optional sheet name. If empty, uses the active sheet.
    """
    args = ["--file_path", file_path, "--text", str(text),
            "--row_idx", str(row_idx), "--column_idx", str(column_idx)]
    if sheet_name:
        args.extend(["--sheet_name", sheet_name])
    return _run_app("excel_app", "excel_set_cell", args)


@tool
def excel_delete_cell(file_path: str, row_idx: int, column_idx: int, sheet_name: str = "") -> str:
    """Delete (clear) a specific cell in an Excel file.

    Args:
        file_path: Path to the Excel file.
        row_idx: Row index (1-based).
        column_idx: Column index (1-based).
        sheet_name: Optional sheet name.
    """
    args = ["--file_path", file_path,
            "--row_idx", str(row_idx), "--column_idx", str(column_idx)]
    if sheet_name:
        args.extend(["--sheet_name", sheet_name])
    return _run_app("excel_app", "excel_delete_cell", args)


@tool
def excel_create_new_file(file_path: str) -> str:
    """Create a new empty Excel file.

    Args:
        file_path: Path for the new Excel file (e.g. '/testbed/data/new.xlsx').
    """
    return _run_app("excel_app", "excel_create_new_file", ["--file_path", file_path])


@tool
def excel_convert_to_pdf(excel_file_path: str, pdf_file_path: str) -> str:
    """Convert an Excel file to PDF.

    Args:
        excel_file_path: Path to the source Excel file.
        pdf_file_path: Path for the output PDF file.
    """
    return _run_app("excel_app", "excel_convert_to_pdf", [
        "--excel_file_path", excel_file_path, "--pdf_file_path", pdf_file_path,
    ])


# ── Word Tools ──────────────────────────────────────────────────────────────


@tool
def word_read_file(file_path: str) -> str:
    """Read the text content of a Word (.docx) file.

    Args:
        file_path: Path to the Word file.
    """
    return _run_app("word_app", "word_read_file", ["--file_path", file_path])


@tool
def word_create_new_file(file_path: str) -> str:
    """Create a new empty Word (.docx) file.

    Args:
        file_path: Path for the new Word file (e.g. '/testbed/data/report.docx').
    """
    return _run_app("word_app", "word_create_new_file", ["--file_path", file_path])


@tool
def word_write_to_file(file_path: str, contents: str, style: str = "pure-text") -> str:
    """Append text content to a Word file.

    Args:
        file_path: Path to the Word file (must already exist).
        contents: The text content to append.
        style: Text style - 'pure-text' (paragraph), 'title', or 'subtitle'.
    """
    return _run_app("word_app", "word_write_to_file", [
        "--file_path", file_path, "--contents", contents, "--style", style,
    ])


@tool
def word_convert_to_pdf(word_file_path: str, pdf_file_path: str) -> str:
    """Convert a Word file to PDF.

    Args:
        word_file_path: Path to the source Word file.
        pdf_file_path: Path for the output PDF file.
    """
    return _run_app("word_app", "word_convert_to_pdf", [
        "--word_file_path", word_file_path, "--pdf_file_path", pdf_file_path,
    ])


# ── PDF Tools ───────────────────────────────────────────────────────────────


@tool
def pdf_read_file(pdf_file_path: str) -> str:
    """Read the text content of a PDF file.

    Args:
        pdf_file_path: Path to the PDF file.
    """
    return _run_app("pdf_app", "pdf_read_file", ["--pdf_file_path", pdf_file_path])


@tool
def pdf_convert_to_image(pdf_file_path: str, image_file_path: str) -> str:
    """Convert a PDF file to an image.

    Args:
        pdf_file_path: Path to the source PDF file.
        image_file_path: Path for the output image file.
    """
    return _run_app("pdf_app", "pdf_convert_to_image", [
        "--pdf_file_path", pdf_file_path, "--image_file_path", image_file_path,
    ])


@tool
def pdf_convert_to_word(pdf_file_path: str, word_file_path: str) -> str:
    """Convert a PDF file to a Word document.

    Args:
        pdf_file_path: Path to the source PDF file.
        word_file_path: Path for the output Word file.
    """
    return _run_app("pdf_app", "pdf_convert_to_word", [
        "--pdf_file_path", pdf_file_path, "--word_file_path", word_file_path,
    ])


# ── OCR Tool ────────────────────────────────────────────────────────────────


@tool
def ocr_recognize_file(file_path: str) -> str:
    """Perform OCR (optical character recognition) on an image file to extract text.

    Args:
        file_path: Path to the image file.
    """
    return _run_app("ocr_app", "ocr_recognize_file", ["--file_path", file_path])


# ── All Tools List ──────────────────────────────────────────────────────────

ALL_TOOLS = [
    # Calendar
    calendar_create_event,
    calendar_delete_event,
    calendar_list_events,
    # Email
    email_send_email,
    email_list_emails,
    email_read_email,
    # Excel
    excel_read_file,
    excel_set_cell,
    excel_delete_cell,
    excel_create_new_file,
    excel_convert_to_pdf,
    # Word
    word_read_file,
    word_create_new_file,
    word_write_to_file,
    word_convert_to_pdf,
    # PDF
    pdf_read_file,
    pdf_convert_to_image,
    pdf_convert_to_word,
    # OCR
    ocr_recognize_file,
]
