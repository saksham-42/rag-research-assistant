import pymupdf, os, sys

def extract_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if not text.strip():
            print(f"Warning: Page {page_num + 1} has no text — possibly scanned.")
            continue

        full_text += text + f"\n[Page {page_num + 1}]\n"

    doc.close()

    if not full_text.strip():
        print(f"Warning: '{pdf_path}' appears to be a scanned PDF. No text extracted.")
    
    filename = os.path.basename(pdf_path).replace(".pdf", ".txt")
    output_path = os.path.join("output", filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved :{output_path}")
    
if __name__ == "__main__":
    pdf_path = sys.argv[1]
    extract_pdf(pdf_path)
