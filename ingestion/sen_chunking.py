import tiktoken, re, sys, os
from nltk.tokenize import sent_tokenize

def get_page_number(tokens, token_position, enc):
    partial_text = enc.decode(tokens[:token_position])
    pages = re.findall(r'\[Page (\d+)\]', partial_text)
    return int(pages[-1]) if pages else 1

def sent_chunker(text, filename, chunk_size = 512, overlap = 50):
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = sent_tokenize(text)

    chunks = []
    chunk_index = 0
    curr_tokens = []
    curr_sen = []

    for sen in sentences:
        sen_tokens = enc.encode(sen)

        if len(curr_tokens)+len(sen_tokens) > chunk_size:
            if curr_tokens:
                chunk_text_str = enc.decode(curr_tokens)
                chunks.append({
                    "text": chunk_text_str,
                    "metadata": {
                        "source": filename,
                        "chunk_index": chunk_index,
                        "page": get_page_number(curr_tokens, 0, enc)
                    }
                })
                chunk_index += 1

                # overlap — keep last few sentences
                last_sen = curr_sen[-1] if curr_sen else ""
                last_tokens = enc.encode(last_sen)
                curr_tokens = last_tokens if len(last_tokens) <= overlap else []
                curr_sen = [last_sen] if curr_tokens else []

        curr_tokens += sen_tokens
        curr_sen.append(sen)

    # last chunk
    if curr_tokens:
        chunks.append({
            "text": enc.decode(curr_tokens),
            "metadata": {
                "source": filename,
                "chunk_index": chunk_index,
                "page": get_page_number(curr_tokens, 0, enc)
            }
        })

    return chunks


if __name__ == "__main__":
    input_path = sys.argv[1]
    text = open(input_path, encoding="utf-8").read()
    filename = os.path.basename(input_path)
    
    chunks = sent_chunker(text, filename)

    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:\n{chunks[0]['text'][:300]}")