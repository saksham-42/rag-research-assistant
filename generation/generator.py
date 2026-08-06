from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from retrieval.fallback import format_fallback, get_fallback_results
from retrieval.hybrid import retrieve, hybrid_retriever
from dotenv import load_dotenv
import os, time

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

PROMPT = """You are a materials science research assistant with deep expertise in metallurgy and materials science.
Answer only using the provided context.
You may combine and synthesize information from multiple retrieved chunks if they discuss the same topic.
Do not introduce information that is not supported by the provided context.
Follow these rules strictly based on what the context contains:
- If the context FULLY answers the question: answer completely and in detail.
- If the context PARTIALLY answers the question: answer using what is available, then clearly state what specific information was not found in the documents.
- If the context contains relevant information, answer using it even if incomplete. Do not refuse the entire question because one detail is missing and then fallback to websearch.
- If the retrieved context contains ANY information related to the question, you MUST answer using that information — even if it is incomplete.NEVER combine a partial answer with the phrase 'I can't find this in the uploaded documents.'That phrase is ONLY for when the context has absolutely nothing relevant.
- When there is absolutely no connection between the context and the question. Reply EXACTLY - 'Not enough information regarding question in uploaded documents' Do NOT use this phrase if you have partial information.

For questions totally unrelated to metallurgy and materials science, reply EXACTLY and ONLY with this phrase, nothing else: 'The question asked is outside of scope of this system'
Always be specific and to the point, keep it technical, always try to include values, temperatures, compositions and mechanisms whenever available and applicable.
Give a detailed, thorough answer — do not summarize or truncate, wherever it is possible.(Mostly try to give detailed answers).
When citing a finding, mention the paper title, page range naturally inline. 
Reference the source paper and page number in the answer.If the chunk retrieved id from the same source and page, reference it at the last, for all those answers.
Do not include a sources list, metadata section, or references section in your answer. Sources are handled separately.
Do not include journal names, volume numbers, or manuscript references in your answer.
Context : {context} 
"""

prompt = ChatPromptTemplate.from_messages([
    ("system",PROMPT), ("human", "{input}")
])

THRESHOLD = 0.7

def fallback(question):
    results, source = get_fallback_results(question)
    ans = format_fallback(results, source)
    return {
        "answer":ans,
        "citations":[],
        "chunks_used":0
    }

def sources(source):
    seen = set()
    ct = 1
    citations = []
    for doc in source:
        src = doc.metadata.get('source')
        if src in seen:
            continue
        seen.add(src)
        fp = doc.metadata.get('first_page', '?')
        lp = doc.metadata.get('last_page', '?')
        title = doc.metadata.get('title', 'Unknown') or os.path.basename(src)
        citations.append({
            "title":title,
            "source": src,
            "first_page":fp,
            "last_page":lp
        })
        ct+=1
    return citations

def document_chain(retriever):
    combine_docs_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

def ask(question, k=5, source = None):
    t0 = time.time()
    result = retrieve(question,k=k,source=source)
    ret_t = time.time()-t0

    score = result["score"]
    print(f"Score: {score:.4f} | Threshold: {THRESHOLD}")
    print(f"Retrieval time: {ret_t:.2f}s")

    if score < THRESHOLD:
        print("Score below threshold")
        return fallback(question)

    t1 = time.time()
    r = hybrid_retriever(k=k,source=source)
    chain = document_chain(r)
    result = chain.invoke({"input": question})
    gen_t = time.time()-t1

    print(f"Generation time : {gen_t:.2f}s")
    print(f"Total time : {gen_t + ret_t:.2f}s")

    answer = result["answer"]
    source = result["context"]
    answer_clean = answer.strip().lower()

    if "information regarding" in answer_clean:
        citation = sources(source)
        print("\nPartial Answer...\n")
        fallback(question) 
        return {
            "answer":answer,
            "citations":citation,
            "chunks_used": len(source)
        }

    elif "outside of scope" in answer_clean:
        return {
            "answer":answer,
            "citations":[],
            "chunks_used": 0
        }
    else :
        citation = sources(source)
        return {
            "answer":answer,
            "citations":citation,
            "chunks_used": len(source)
        }

if __name__ == "__main__":
    query = "How does Cr content influence corrosion resistance in CoNiAl HEAs?"
    ask(query, k=5)