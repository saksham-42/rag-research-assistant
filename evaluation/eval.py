import os, time
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from retrieval.hybrid import retrieve

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
    "Compare nanograined Ti-6Al-4V and martensitic steel in terms of strengthening mechanisms.",
    "Compare additive manufacturing approaches and conventional processing routes discussed in the corpus.",
    "Which alloy system in the corpus appears most promising for aerospace applications, and why?",
    "If maximum hardness is the design objective, which material and processing route would you choose from the corpus?",
    "What trade-offs between strength and ductility are observed across the alloy systems in the papers?",
    "What manufacturing methods are used for High Entropy Alloys?",
    "What phases are present in Ti-6Al-4V after inert gas condensation processing?",
    "What are the core effects proposed in High Entropy Alloy theory?",
    "What are the deformation mechanisms in CoCrNi medium entropy alloys?",
    "How does Al content affect the phase stability of HEAs?",
    "What is the effect of Nb addition on the mechanical properties of FeCoNiAl HEAs?",
    "How do HEAs perform in aeroengine applications compared to conventional superalloys?",
    "What phases form in CoCrNiAlCu high entropy alloys?",
    "How does temperature affect the yield strength of HEAs?",
    "What is the role of the high entropy effect in phase diagram prediction of HEAs?",
    "How does Cr content influence corrosion resistance in CoNiAl HEAs?",
    "What are the strengthening mechanisms in FeCoNi based HEAs?",
    "How does grain size affect mechanical properties of CoCrNiAl alloys?",
    "What is the effect of shot peening on fatigue life of Ti-6Al-4V?",
    "How does the omega phase form in titanium alloys and how does it affect mechanical properties?",
    "What are the defects formed during additive manufacturing of TiAl alloys?",
    "How does inert atmosphere processing affect microstructure of Ti-6Al-4V?",
    "What is the effect of heat treatment on omega phase stability in Ti alloys?",
    "What is the effect of build orientation on mechanical properties in additively manufactured TiAl alloys?",
    "What is the effect of Sc and Zr additions on mechanical properties of Al-Mg-Si alloys made by additive manufacturing?",
    "How does aging treatment affect precipitation hardening in Al-Mg-Si alloys?",
    "What is the effect of Zn content on corrosion behavior of Al-Zn alloys?",
    "How does thermo-mechanical processing affect microstructure of Al-Zn-Mg alloys?",
    "What precipitates form during aging of Al-Mg-Si alloys?",
    "How does solution treatment temperature affect mechanical properties of Al-Mg-Si-Mn-Sc-Zr alloys?",
    "What is the creep-fatigue crack initiation mechanism in modified 12% Cr steel?",
    "How does nitrogen content affect martensite transformation in martensitic steels?",
    "What are the deformation mechanisms in dual phase steels?",
    "How does grain boundary cavitation contribute to creep damage in Cr steels?",
    "What is the effect of plastic slip accumulation on fatigue crack initiation in 12% Cr steel?",
    "Compare the yield strength of HEAs versus Ti-6Al-4V at elevated temperatures.",
    "What additive manufacturing challenges are common between TiAl and Al-Mg-Si alloys?",
    "How does precipitation hardening differ between aluminum alloys and high entropy alloys?",
    "What is the exact composition of the CoCrNiAlCu high entropy alloy studied in the corpus?",
    "What annealing temperature was used to achieve peak hardness in the nanograined Ti-6Al-4V study?",
    "What was the reported grain size of the nanograined Ti-6Al-4V before and after annealing?",
    "What aging temperature and duration produced maximum precipitation hardening in the Al-Mg-Si alloy?",
    "What was the reported fracture toughness of the dual phase steel studied in the corpus?",
    "What characterization techniques were used to study the omega phase in titanium alloys?",
    "What post-processing steps were applied to TiAl alloys after additive manufacturing?",
    "What crack initiation sites were identified in the 12% Cr steel fatigue study?",
    "What is the reported density advantage of TiAl alloys over conventional nickel superalloys?",
    "What corrosion testing method was used to evaluate the Al-Zn-Mg-Cu alloy?",
    "Why does the omega phase embrittle titanium alloys despite increasing hardness?",
    "Why does increasing Al content beyond a certain threshold destabilize the FCC phase in HEAs?",
    "Why does grain boundary cavitation preferentially initiate at specific boundary types in Cr steels?",
    "Why does shot peening introduce compressive residual stresses and how do these suppress fatigue crack initiation?",
    "Why does the high entropy effect not always guarantee a single-phase solid solution microstructure?",
    "What role does heat treatment play across all alloy systems studied in the corpus?",
    "How is oxidation resistance addressed across the HEA and titanium alloy papers in the corpus?",
    "What characterization techniques are most commonly used across all papers in the corpus?",
    "How do the failure mechanisms differ between the steel and titanium alloy papers?",
    "What processing-microstructure-property relationships are consistently observed across all alloy systems?",
    "Compare the effect of grain refinement on strength and ductility in HEAs versus dual phase steels.",
    "Compare the role of Al additions in Ti-Al alloys versus Al additions in HEAs.",
    "How does creep resistance compare between the 12% Cr steel and HEAs discussed in the corpus?",
    "Compare the corrosion mechanisms reported for Al-Zn alloys versus Cr-containing HEAs.",
    "Compare the effect of secondary phase precipitation on mechanical properties in Al-Mg-Si alloys and HEAs.",
    "Based on the corpus, which microstructural feature has the strongest influence on yield strength across all alloy systems?",
    "If you had to design an alloy for simultaneous high strength, corrosion resistance, and high temperature stability, what would the corpus evidence support?",
    "How do the strengthening contributions of solid solution hardening, precipitation hardening, and grain boundary strengthening compare quantitatively across the alloy systems?",
    "What does the corpus collectively suggest about the relationship between processing route and fatigue life across titanium alloys and steels?",
    "Based on evidence across all papers, what are the fundamental limitations of additive manufacturing for producing high performance metallic alloys?",
    "What is the effect of hydrogen embrittlement on titanium alloys used in aerospace?",
    "How does friction stir welding affect the microstructure of Al-Mg-Si alloys?",
    "What are the radiation damage mechanisms in high entropy alloys for nuclear applications?",
    "How does thermodynamic CALPHAD modelling compare to experimental phase diagram results for HEAs?",
    "What are the fatigue crack propagation rates in additively manufactured Ti-6Al-4V under variable amplitude loading?",
]

TOP_K = 3

def is_relevant(question, chunk_text):
    prompt = f"""You are evaluating a RAG retrieval system for materials science.

Question: {question}

Retrieved chunk:
{chunk_text[:1500]}

Does this chunk contain information that is useful for answering the question, even partially?
Reply with only one word: Yes or No."""

    response = judge.invoke(prompt)
    return "yes" in response.strip().lower()

def run():
    hits = 0
    precision_scores = []
    worst = []

    for i, question in enumerate(QUESTIONS):
        result = retrieve(question, k=TOP_K)
        chunks = result["documents"]

        relevant_flags = []
        for chunk in chunks[:TOP_K]:
            relevant = is_relevant(question, chunk.page_content)
            relevant_flags.append(relevant)
            time.sleep(1.0)

        hit = any(relevant_flags)
        precision = sum(relevant_flags) / TOP_K

        if hit:
            hits += 1
        precision_scores.append(precision)
        worst.append((precision, i + 1, question))

        status = "HIT" if hit else "MISS"
        print(f"[{i+1:02d}] {status} | P@3={precision:.2f} | {question[:60]}")

    total = len(QUESTIONS)
    hit_rate = hits / total
    mean_precision = sum(precision_scores) / total

    print(f"\n{'='*50}")
    print(f"Total questions   : {total}")
    print(f"Hit Rate @3       : {hit_rate:.1%}")
    print(f"Mean Precision @3 : {mean_precision:.1%}")
    print(f"{'='*50}")

    worst_5 = sorted(worst, key=lambda x: x[0])[:5]
    print(f"\n5 WORST QUESTIONS (use these on Day 5):")
    for score, num, q in worst_5:
        print(f"  Q{num} | P@3={score:.2f} | {q[:70]}")

if __name__ == "__main__":
    run()