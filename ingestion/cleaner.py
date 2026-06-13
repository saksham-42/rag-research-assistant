import re, sys

REPLACEMENTS = {
    '\u2013': '-',        # en-dash
    '\u2014': ' - ',      # em-dash
    '\u201c': '"',        # left curly quote
    '\u201d': '"',        # right curly quote
    '\u2019': "'",        # right curly apostrophe
    '\u02c7': '',         # caron
    '\u25e6': 'degrees',  # circle/degree symbol
}

def fix_unicode(text):
    for char, replacement in REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text

def remove_headers_footers(text):
    # remove the headers starting with metals /date/date/
    text = re.sub(r'Metals \d{4}, \d+, \d+', '', text)
    # remove the pages(1 of 22, etc.)
    text = re.sub(r'\d+ of \d+', '', text)
    #remove URLs
    text = re.sub(r'https?://\S+', '', text)
    #remove '.'
    text = re.sub(r'^\s*\.\s*$', '', text)
    return text

def fix_spacing(text):
    # fix the spaces between texts
    text = re.sub(r' {2,}', ' ', text)
    # fix the gap between paragraphs(too many lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # fix trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text

def fix_hyphenation(text):
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

def fix_line_breaks(text):
    return re.sub(r'([a-zA-Z,])\n([a-zA-Z])', r'\1 \2', text)

def fix_section_headings(text):
    text = re.sub(r'(\d+\.\s+[A-Z][a-zA-Z\s]+?)([A-Z][a-z])', r'\1\n\2', text)
    return text

def remove_citation(text):
    text = re.sub(r'(Received|Revised|Accepted|Published|Correspondence|Copyright|Licensee|Academic Editor):.*', '', text)
    text = re.sub(r'^(Received|Revised|Accepted|Published|Correspondence|Copyright|Licensee)\s*$', '', text, flags=re.MULTILINE)
    return text

def clean_text(text):
    text = fix_unicode(text)
    text = remove_citation(text)
    text = remove_headers_footers(text)
    text = fix_hyphenation(text)
    text = fix_section_headings(text)
    text = fix_line_breaks(text)
    text = fix_spacing(text)
    return text.strip()

if __name__ == "__main__":
    input_path = sys.argv[1]

    text = open(input_path, encoding="utf-8").read()
    cleaned = clean_text(text)

    output_path = sys.argv[2]

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(cleaned)

    print(f"Done. Saved to {output_path}") 