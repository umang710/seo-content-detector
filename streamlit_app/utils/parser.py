import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

def scrape_and_parse_url(url):
    """Scrape and parse a single URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        time.sleep(1)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else ""
        
        # Extract main content
        selectors = ['article', 'main', '.content', '.post-content', 'p']
        body_text = ""
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text().strip() for elem in elements])
                if len(text) > 100:
                    body_text = text
                    break
        
        if not body_text:
            body_text = soup.get_text()
        
        body_text = ' '.join(body_text.split())
        word_count = len(body_text.split())
        
        return title, body_text, word_count
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "", "", 0