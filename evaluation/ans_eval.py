import os, time
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from generation.generator import ask

load_dotenv()

judge = GoogleGenerativeAI(
    model="models/gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
    )

QUESTIONS = [
    "What manufacturing routes are used for advanced metallic materials in the papers, and what are their advantages?",
    "What hardness was achieved by the nanograined Ti-6Al-4V alloy after annealing at 400°C?",
    "What surface hardness improvement was reported after severe warm shot peening of Ti-6Al-4V?",
    "What yield strength and tensile strength were reported for the Al-Mg-Si alloy studied in the corpus?",
    "Why does hardness increase after annealing at 400°C but decrease after annealing at 800°C in nanograined Ti-6Al-4V?",
    "Why does severe warm shot peening improve the mechanical performance of Ti-6Al-4V?",
    "Why do High Entropy Alloys often exhibit superior high-temperature properties compared to conventional alloys?",
    "Summarize the major strengthening mechanisms discussed across the alloy systems in the corpus.",
    "Summarize the role of microstructure engineering in improving mechanical properties across the studied alloys.",
    "Compare the manufacturing routes used for High Entropy Alloys and Ti-6Al-4V alloys.",
    "What are the core effects proposed in High Entropy Alloy theory?",
    "What phases are present in Ti-6Al-4V after inert gas condensation processing?",
    "How does Al content affect the phase stability of HEAs?",
    "What is the effect of Nb addition on the mechanical properties of FeCoNiAl HEAs?",
    "How do HEAs perform in aeroengine applications compared to conventional superalloys?",
    "How does temperature affect the yield strength of HEAs?",
    "What is the effect of shot peening on fatigue life of Ti-6Al-4V?",
    "How does the omega phase form in titanium alloys and how does it affect mechanical properties?",
    "What are the defects formed during additive manufacturing of TiAl alloys?",
    "How does aging treatment affect precipitation hardening in Al-Mg-Si alloys?",
    "What precipitates form during aging of Al-Mg-Si alloys?",
    "What is the creep-fatigue crack initiation mechanism in modified 12% Cr steel?",
    "What are the deformation mechanisms in dual phase steels?",
    "How does grain boundary cavitation contribute to creep damage in Cr steels?",
    "What is the effect of heat treatment on omega phase stability in Ti alloys?",
    "What role does heat treatment play across all alloy systems studied in the corpus?",
    "Why does the omega phase embrittle titanium alloys despite increasing hardness?",
    "How does Cr content influence corrosion resistance in CoNiAl HEAs?",
    "What are the strengthening mechanisms in FeCoNi based HEAs?",
    "What is the effect of Zn content on corrosion behavior of Al-Zn alloys?",
]

def evaluate_answer(question, answer, chunks):
    context = "\n\n".join([c.page_content for c in chunks[:5]])

    faithfulness_prompt = f"""You are evaluating a RAG system answer for faithfulness.

Question: {question}

Retrieved context:
{context}

Answer given:
{answer}

Score the answer from 1 to 5 for faithfulness — is every claim in the answer supported by the retrieved context?
1 = answer contains claims not in context at all
3 = answer mostly supported but has some unsupported claims
5 = every claim is fully supported by the context

Reply with only a single digit 1, 2, 3, 4, or 5."""

    relevance_prompt = f"""You are evaluating a RAG system answer for relevance.

Question: {question}

Answer given:
{answer}

Score the answer from 1 to 5 for relevance — does the answer actually address what was asked?
1 = answer is completely off-topic
3 = answer partially addresses the question
5 = answer directly and completely addresses the question

Reply with only a single digit 1, 2, 3, 4, or 5."""

    try:
        f_response = judge.invoke(faithfulness_prompt)
        time.sleep(2)
        r_response = judge.invoke(relevance_prompt)
        time.sleep(2)

        faithfulness = int(f_response.strip()[0])
        relevance = int(r_response.strip()[0])
    except Exception:
        faithfulness = 0
        relevance = 0

    return faithfulness, relevance

def run():
    total_faith = 0
    total_rel = 0
    scored = 0
    worst = []

    for i, question in enumerate(QUESTIONS):
        result = ask(question, k=5)
        answer = result.get("answer", "")

        from retrieval.hybrid import retrieve
        chunks = retrieve(question, k=3)["documents"]

        faith, rel = evaluate_answer(question, answer, chunks)
        total_faith += faith
        total_rel += rel
        scored += 1
        worst.append((faith + rel, i + 1, question))

        print(f"[{i+1:02d}] F={faith} R={rel} | {question[:65]}")

    print(f"\n{'='*50}")
    print(f"Questions scored     : {scored}")
    print(f"Avg Faithfulness     : {total_faith/scored:.2f} / 5")
    print(f"Avg Relevance        : {total_rel/scored:.2f} / 5")
    print(f"{'='*50}")

    worst_5 = sorted(worst, key=lambda x: x[0])[:5]
    print(f"\n5 WORST ANSWERS:")
    for score, num, q in worst_5:
        print(f"  Q{num} | Combined={score} | {q[:70]}")

if __name__ == "__main__":
    run()