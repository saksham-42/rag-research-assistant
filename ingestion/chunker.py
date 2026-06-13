import tiktoken, os, sys, re

def get_page_number(tokens, token_position, enc):
    partial_text = enc.decode(tokens[:token_position])
    pages = re.findall(r'\[Pages (\d+)\]', partial_text)
    return int(pages[-1]) if pages else 1

def chunking(text, filename, chunk_size=512, overlap = 50):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    chunk_index=0
    start=0

    while start<len(tokens):
        end = start+chunk_size
        chunk_tokens = tokens[start : end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append({
            "text" : chunk_text,
            "metadata" :{
                "source":filename,
                "chunk_index" : chunk_index,
                "page" : get_page_number(tokens,start,enc)
            }
        })

        chunk_index+=1
        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":
    input_path = sys.argv[1]
    text = open(input_path, encoding="utf-8").read()
    filename = os.path.basename(input_path)

    chunks = chunking(text, filename)

    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk preview:\n{chunks[0]['text'][:300]}")