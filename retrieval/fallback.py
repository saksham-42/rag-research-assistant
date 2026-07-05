import requests, time, os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

load_dotenv()

def search_semantic(query, limit=3, retries=3):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":query,
        "limit":limit,
        "fields":"title,authors,year,abstract,externalIds"
    }
    api_key = os.getenv("SEMANTIC_API_KEY")
    if api_key:
        headers = {"x-api-key": api_key}
    else:
        headers = {}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"Rate limited — waiting {wait}s before retry...")
                time.sleep(wait)
                continue
                
            response.raise_for_status()
            data = response.json()
            results = []
            for paper in data.get("data", []):
                title = paper.get("title", "Unknown")
                authors = ", ".join(a["name"] for a in paper.get("authors", [])[:3])
                year = paper.get("year", "N/A")
                abstract = paper.get("abstract") or "No abstract available."
                abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
                doi = paper.get("externalIds", {}).get("DOI", "")
                link = f"https://doi.org/{doi}" if doi else "https://www.semanticscholar.org"
                results.append({
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "link": link
                })
            return results
            
        except Exception as e:
            print(f"Semantic Scholar API error: {e}")
            if attempt < retries - 1:
                time.sleep(1)
            
    return []

def search_arxiv(query, limit=3):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        print(f"Found entries : {len(entries)}")

        results = []
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
            link = entry.find("atom:id", ns).text.strip()
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)[:3]]
            results.append({
                "title": title,
                "authors": ", ".join(authors),
                "year": "N/A",
                "abstract": abstract,
                "link": link
            })
        return results
    except Exception as e:
        print(f"ArXiv API error: {e}")
        return []

def get_fallback_results(query):
    results = search_semantic(query)
    if results:
        return results, "Semantic Scholar"
    print("Trying ArXiv...")
    results = search_arxiv(query)
    return results, "ArXiV"

def format_fallback(results, source="Semantic Scholar"):
    lines = [f"Not found in your uploaded papers. Found on {source}:\n"]
    for i, r in enumerate(results):
        lines.append(f"[{i+1}] {r['title']} ({r['year']})")
        lines.append(f" Authors: {r['authors']}")
        lines.append(f" {r['abstract']}")
        lines.append(f" Link: {r['link']}\n")
    return "\n".join(lines)