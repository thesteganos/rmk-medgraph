# knowledge_pipeline.py

import os
import requests
from bs4 import BeautifulSoup
import json
import sys
from dotenv import load_dotenv
from urllib.parse import urljoin, quote_plus, urlparse, urlunparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

TRUSTED_SOURCES = {
    "WHO_Healthy_Diet": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "NIH_Supplements_Factsheet": "https://ods.od.nih.gov/factsheets/WeightLoss-HealthProfessional/"
}
TERMS_OF_INTEREST = [
    "semaglutide for weight loss",
    "intermittent fasting science",
    "metabolic syndrome risk factors",
    "benefits of regular physical activity",
    "understanding cholesterol levels",
    "managing type 2 diabetes with diet",
    "common nutritional deficiencies",
]
TRUSTED_SEARCH_SITES = {
    "Mayo Clinic Search": "https://www.mayoclinic.org/search/search-results?q={query}",
    "NIH Search": "https://www.nih.gov/search/results?keys={query}",
    "CDC Search": "https://www.cdc.gov/search/?query={query}",
    "MedlinePlus Search": "https://medlineplus.gov/search/all?query={query}",
    "Cleveland Clinic Search": "https://my.clevelandclinic.org/search?q={query}",
}

PENDING_REVIEW_FILE = "pending_review.jsonl"
PROCESSED_URLS_LOG = "processed_urls.log"

def get_processed_urls():
    if not os.path.exists(PROCESSED_URLS_LOG): return set()
    with open(PROCESSED_URLS_LOG, "r", encoding='utf-8') as f:
        return set(line.strip() for line in f)

def log_processed_url(url):
    with open(PROCESSED_URLS_LOG, "a", encoding='utf-8') as f:
        f.write(url + "\n")

def get_article_text(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('article') or soup.find(id='main-content') or soup.find(role='main') or soup.body
        if main_content:
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            return ' '.join(p.get_text(strip=True) for p in main_content.find_all('p'))
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article {url}: {e}")
        return None

def scrape_search_results(search_url: str, base_url: str) -> list[str]:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        for tag in soup.select('h2 a, h3 a, h4 a'):
            if 'href' in tag.attrs:
                full_url = urljoin(base_url, tag['href'])
                if full_url not in links:
                    links.append(full_url)
            if len(links) >= 3: break
        return links
    except requests.exceptions.RequestException as e:
        print(f"Error scraping search results {search_url}: {e}")
        return []

def generate_full_search_url_and_base(search_template: str, query: str) -> tuple[str, str]:
    encoded_query = quote_plus(query)
    full_search_url = search_template.format(query=encoded_query)
    parsed_search_url = urlparse(full_search_url)
    base_url = urlunparse((parsed_search_url.scheme, parsed_search_url.netloc, '', '', '', ''))
    return full_search_url, base_url

def main():
    print("--- Starting Automated Knowledge Pipeline ---")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("LLM_MODEL")
    if not google_api_key or not model_name:
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY and LLM_MODEL must be set in .env")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=google_api_key)
    prompt = ChatPromptTemplate.from_template(
        """You are a highly specialized medical knowledge extraction system. Your task is to analyze the provided medical article text and formulate a specific, answerable question that the text addresses, along with a comprehensive yet concise answer derived exclusively from the provided text.

**Instructions:**
1.  **Specificity:** The question should be specific and target a key piece of information or conclusion presented in the article. Avoid overly broad questions.
2.  **Fidelity:** The answer MUST be based ONLY on the information explicitly stated or clearly implied within the provided 'ARTICLE TEXT'. Do not introduce external knowledge or assumptions.
3.  **Clarity & Conciseness:** The answer should be clear, easy to understand, and as concise as possible while still capturing the essential information needed to address the question thoroughly.
4.  **JSON Output:** Your output MUST be a single, valid JSON object. This object should contain exactly two keys: "question" and "answer". Do not include any explanatory text, markdown formatting, or any other content outside of this JSON structure.

**Example of a good question:** "What were the primary cardiovascular outcomes observed in patients with Type 2 Diabetes continuously using Metformin, according to the 2023 Journal of Internal Medicine study?"
**Example of a less ideal question:** "What does the article say about diabetes?"

ARTICLE TEXT:
---
{article_text}
---

JSON OUTPUT:
"""
    )
    extraction_chain = prompt | llm | StrOutputParser()

    processed_urls = get_processed_urls()
    new_articles_to_process = []
    
    for term in TERMS_OF_INTEREST:
        for site_name, search_template in TRUSTED_SEARCH_SITES.items():
            search_url, base_url = generate_full_search_url_and_base(search_template, term)
            article_urls = scrape_search_results(search_url, base_url)
            for url in article_urls:
                if url not in processed_urls:
                    article_text = get_article_text(url)
                    if article_text:
                        new_articles_to_process.append({"text": article_text, "url": url})
                        processed_urls.add(url) # Avoid re-processing in same run
    
    if not new_articles_to_process:
        print("\nNo new articles found.")
        return

    print(f"\n--- Found {len(new_articles_to_process)} new articles. Processing... ---")
    
    for article in new_articles_to_process:
        try:
            result_str = extraction_chain.invoke({"article_text": article["text"]})
            proposition = json.loads(result_str)
            proposition['source_url'] = article["url"]
            with open(PENDING_REVIEW_FILE, "a", encoding='utf-8') as f:
                f.write(json.dumps(proposition) + "\n")
            log_processed_url(article["url"])
            print(f"--> Proposition saved for review from {article['url']}")
        except Exception as e:
            print(f"--> FAILED to process {article['url']}. Error: {e}")
    
    print("\n--- Pipeline run complete. ---")

if __name__ == "__main__":
    main()