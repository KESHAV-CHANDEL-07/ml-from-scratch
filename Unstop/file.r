from docx import Document

def replace_text_preserve_format(container, old_text, new_text):
    """
    Replace text in every run (keeps alignment, font, and spacing).
    Works for paragraphs, tables, headers, and footers.
    """
    # Replace in normal paragraphs
    for paragraph in container.paragraphs:
        for run in paragraph.runs:
            if old_text in run.text:
                run.text = run.text.replace(old_text, new_text)

    # Replace inside tables too
    for table in container.tables:
        for row in table.rows:
            for cell in row.cells:
                replace_text_preserve_format(cell, old_text, new_text)


def update_docx(input_path, output_path):
    doc = Document(input_path)

    replacements = {
        "Madhaw Kumar Bagadia": "Keshav Chandel",
        "MADHAW KUMAR BAGADIA": "KESHAV CHANDEL",
        "23BEC058": "23BEC053"
    }

    # Replace in document body
    for old, new in replacements.items():
        replace_text_preserve_format(doc, old, new)

    # Replace in headers and footers for each section
    for section in doc.sections:
        header = section.header
        footer = section.footer
        for old, new in replacements.items():
            replace_text_preserve_format(header, old, new)
            replace_text_preserve_format(footer, old, new)

    # Save updated file
    doc.save(output_path)
    print(f"✅ Updated file saved as: {output_path}")


# ---------- Example Usage ----------
if __name__ == "__main__":
    input_file = "MATLAB_File[1].docx"          # Original file path
    output_file = "MATLAB_File_Updated_Final.docx"  # Output file path
    update_docx(input_file, output_file)
