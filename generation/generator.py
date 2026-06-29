from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)

PROMPT = """You are a materials science research assistant with deep expertise in metallurgy and materials science.
Answer only using the provided context from the research papers. If you don't get the answer in the context, then say exactly - 'I can't find this in the uploaded documents'.
For totally off topic questions, unrelated to metallurgy and materials science DO NOT provide any references and sources, then reply exactly - 'The question asked is outside of scope of this system.'
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

def document_chain(retriever):
    combine_docs_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

def ask(question, retriever):
    chain = document_chain(retriever)
    result = chain.invoke({"input": question})
    
    answer = result["answer"]
    sources = result["context"]

    print(f"\nAnswer:\n{answer}")
    
    if "outside of scope " in answer.lower() or "can't find this" in answer.lower():
        return result

    print(f"\nSources:")
    seen = set()
    ct = 1
    for i, doc in enumerate(sources):
        src = doc.metadata.get('source')
        if src in seen:
            continue
        seen.add(src)
        fp = doc.metadata.get('first_page', '?')
        lp = doc.metadata.get('last_page', '?')
        title = doc.metadata.get('title', 'Unknown') or os.path.basename(src)
        print(f"  [{ct}] {title} — Pages [{fp}-{lp}] — {src}")
        ct+=1
    
    return result

if __name__ == "__main__":
    from retrieval.hybrid import hybrid_retriever
    query = "What is the capital of France?"
    r = hybrid_retriever(k=5)
    ask(query, r)