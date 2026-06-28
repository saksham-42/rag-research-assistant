import re
from collections import defaultdict

REPLACEMENTS = {
    '\u2013': '-',        # en-dash
    '\u2014': ' - ',      # em-dash
    '\u201c': '"',        # left curly quote
    '\u201d': '"',        # right curly quote
    '\u2019': "'",        # right curly apostrophe
    '\u02c7': '',         # caron
    '\u25e6': 'degrees',  # circle/degree symbol
}

BODY_START = re.compile(
    r'''
    ^\s*
    (?:\d+(?:\.\d+)*\.?\s*)?
    (
        Introduction|
        Background|
        Overview|
        Literature\s+Review|
        Related\s+Work|
        Materials?\s+and\s+Methods?|
        Experimental(?:\s+Procedure|\s+Methods?)?|
        Methodology|
        Methods?|
        Materials?
    )
    \b
    ''',
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)

CONCLUSION = re.compile(
    r'''
    ^\s*
    (?:\d+(?:\.\d+)*\.?\s*)?
    (
        Conclusion|
        Conclusions|
        Summary\s+and\s+Conclusions|
        Concluding\s+Remarks|
        Conclusions\s+and\s+Outlook|
        Conclusions\s+and\s+Future\s+Work
    )
    \b
    ''',
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)

REST = re.compile(
    r'''
    ^\s*
    (?:
        \d+(?:\.\d+)*[.):]?\s*
    )?
    (
        References?|
        Bibliograph(?:y|ies)?|
        Acknowledg(?:e)?ments?|
        Appendix|
        Appendices|
        Supplementary.*|
        Funding.*|
        Declaration.*|
        CRediT.*|
        Author\s+Contributions?.*|
        Data\s+Availability.*|
        Conflict.*|
        Competing.*
    )
    \s*:?\s*$
    ''',
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)

INTRODUCTION = re.compile(r'^\s*(?:\d+\.?\s*)?Introduction\s*\b', re.MULTILINE | re.IGNORECASE)

def fix_unicode(text):
    for char, replacement in REPLACEMENTS.items():
        text = text.replace(char, replacement)
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    return text

def remove_headers_footers(text):
    text = re.sub(r'Metals \d{4}, \d+, \d+', '', text)
    text = re.sub(r'\d+ of \d+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'doi:\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
    text = re.sub(r'ORCID\s*:?\s*\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\.\s*$', '', text)
    text = re.sub(r'^\s*\*?\s*Corresponding author.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*E-mail address:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Contents lists available.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*journal homepage:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Available online.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*©.*Elsevier.*$', '', text, flags=re.MULTILINE)
    return text

def fix_spacing(text):
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text

def fix_hyphenation(text):
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)

def fix_line_breaks(text):
    text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z,])\n([a-zA-Z])', r'\1 \2', text)
    return text

def fix_section_headings(text):
    text = re.sub(r'(\d+\.\s+[A-Z][a-zA-Z\s]+?)([A-Z][a-z])', r'\1\n\2', text)
    return text

def remove_citation(text):
    text = re.sub(r'(Citation|Received|Revised|Accepted|Published|Correspondence|Copyright|Licensee|Academic Editor):.*', '', text)
    text = re.sub(r'^.*\b(Citation|Received|Revised|Accepted|Published|Correspondence|Copyright|Licensee)\b.*$', '', text, flags=re.MULTILINE)
    return text

def clean_text(text):
    text = text.replace('\xad', '')
    text = fix_unicode(text)
    text = remove_citation(text)
    text = remove_headers_footers(text)
    text = fix_section_headings(text)
    text = fix_hyphenation(text)
    text = fix_line_breaks(text)
    text = fix_spacing(text)
    return text.strip()
    

def clean_documents(docs):
    grouped = defaultdict(list)
    for d in docs:
        grouped[d.metadata["source"]].append(d)

    cleaned = []
    for source, pages in grouped.items():
        pages.sort(key=lambda d: d.metadata["page"])

        hit_intro = False

        for page in pages:
            text = page.page_content

            if not hit_intro:
                intro = INTRODUCTION.search(text)
                if intro:
                    title = page.metadata.get('title', '')
                    text = title + '\n' + text[intro.start():]
                    hit_intro = True
                elif page.metadata['page']==0:
                    continue
                else: 
                    hit_intro=True

            rest = REST.search(text)
            if rest:
                text = text[:rest.start()]
                text = clean_text(text)
                if text:
                    page.page_content = text
                    cleaned.append(page)
                break

            text = clean_text(text)
            if text:
                page.page_content = text
                cleaned.append(page)

    return cleaned

if __name__ == "__main__":
    from ingestion.extractor import load_pdfs

    docs = load_pdfs()
    cleaned = clean_documents(docs)

    print(f"Pages before cleaning: {len(docs)}")
    print(f"Pages after cleaning: {len(cleaned)}")
    print(cleaned[0].page_content)