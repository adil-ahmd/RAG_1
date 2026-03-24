import json
import time
import logging
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pages to scrape
START_URLS = [
    "https://zatca.gov.sa/en/HelpCenter/guidelines/Pages/default.aspx"
]

MAX_PAGES = 30

def create_driver():
    """
    Create Chrome browser driver
    """
    options = Options()
    options.add_argument("--headless")           # Run without opening browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    # Pretend to be a real browser
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
    # Check if chromedriver is in path or use webdriver-manager
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        logger.warning(f"Could not use webdriver-manager: {e}. Trying default.")
        driver = webdriver.Chrome(options=options)
        
    return driver

def wait_for_page_load(driver, timeout=15):
    """
    Wait for JavaScript to finish loading content
    """
    try:
        # Wait for download icons to appear
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "img[src*='download.svg']")
            )
        )
        logger.info("✅ Page content loaded")
        return True
    except Exception:
        logger.warning("⚠️ Download icons not found, page may still have content")
        return False

def extract_pdf_links(driver, page_url):
    """
    Extract all PDF links from current page after JS loads
    """
    pdf_links = []
    
    # Find all anchor tags containing download icon
    anchors = driver.find_elements(
        By.CSS_SELECTOR,
        "a:has(img[src*='download.svg'])"
    )
    
    # Fallback if :has() not supported
    if not anchors:
        # Find all download icons, then get parent <a> tag
        icons = driver.find_elements(
            By.CSS_SELECTOR,
            "img[src*='download.svg']"
        )
        anchors = []
        for icon in icons:
            try:
                parent = icon.find_element(By.XPATH, "./ancestor::a[1]")
                anchors.append(parent)
            except:
                continue
    
    for anchor in anchors:
        try:
            href = anchor.get_attribute("href")
            
            if not href:
                continue
            
            # Make absolute URL
            full_url = urljoin(page_url, href)
            
            if not full_url.startswith("http"):
                continue
            
            # Get document title
            # Try to find title from nearby text
            try:
                title = anchor.find_element(
                    By.XPATH,
                    "./preceding-sibling::*[1]"
                ).text.strip()
            except:
                title = ""
            
            # Fallback to filename
            if not title:
                title = href.split("/")[-1]\
                    .replace("%20", " ")\
                    .replace(".pdf", "")\
                    .strip()
            
            pdf_links.append({
                "url": full_url,
                "title": title,
                "source_page": page_url,
                "category": detect_category(page_url)
            })
            
            logger.info(f"  Found: {title} → {full_url}")
        
        except Exception as e:
            logger.warning(f"Error extracting link: {e}")
    
    return pdf_links

def get_subpages(driver, page_url):
    """
    Find subpage links to follow
    """
    subpages = []
    
    links = driver.find_elements(By.CSS_SELECTOR, "a[href*='Pages']")
    
    for link in links:
        href = link.get_attribute("href")
        if href and "zatca.gov.sa" in href and href != page_url:
            subpages.append(href)
    
    return list(set(subpages))

def detect_category(source_url):
    if "guidelines" in source_url.lower():
        return "Guidelines"
    elif "rulesregulations" in source_url.lower():
        return "Rules & Regulations"
    elif "mediacenter" in source_url.lower():
        return "Publications"
    return "Other"

def scrape_all_pages():
    """
    Main scraping function
    Visits all pages, waits for JS, extracts PDF links
    """
    driver = create_driver()
    all_documents = []
    visited_pages = set()
    pages_to_visit = list(START_URLS)
    pages_scraped = 0
    
    try:
        while pages_to_visit and pages_scraped < MAX_PAGES:
            page_url = pages_to_visit.pop(0)
            
            # Skip already visited
            if page_url in visited_pages:
                continue
            
            pages_scraped += 1
            visited_pages.add(page_url)
            logger.info(f"\n📄 Visiting ({pages_scraped}/{MAX_PAGES}): {page_url}")
            
            try:
                # Load page in real browser
                driver.get(page_url)
                
                # Wait for JavaScript to load content
                wait_for_page_load(driver, timeout=10)
                
                # Extra wait for dynamic content
                time.sleep(2)
                
                # Extract PDF links (now visible after JS ran)
                pdf_links = extract_pdf_links(driver, page_url)
                all_documents.extend(pdf_links)
                logger.info(f"  Found {len(pdf_links)} PDFs")
                
                # Find subpages to follow
                subpages = get_subpages(driver, page_url)
                for subpage in subpages:
                    if subpage not in visited_pages:
                        pages_to_visit.append(subpage)
                        logger.info(f"  Queued subpage: {subpage}")
                
                # Polite delay between pages
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error scraping {page_url}: {e}")
    
    finally:
        driver.quit()
    
    # Remove duplicates
    seen_urls = set()
    unique_documents = []
    for doc in all_documents:
        if doc['url'] not in seen_urls:
            seen_urls.add(doc['url'])
            unique_documents.append(doc)
    
    return unique_documents

def main():
    logger.info("="*60)
    logger.info("🕷️  ZATCA Selenium Scraper Starting")
    logger.info("="*60)
    
    # Scrape all pages
    documents = scrape_all_pages()
    
    # Ensure crawler directory exists
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "zatca_documents.json")

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    logger.info(f"\n✅ Complete!")
    logger.info(f"📊 Total PDFs found: {len(documents)}")
    logger.info(f"💾 Saved to: {output_file}")
    
    # Print summary
    categories = {}
    for doc in documents:
        cat = doc['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    logger.info("\n📊 By Category:")
    for cat, count in categories.items():
        logger.info(f"  • {cat}: {count} documents")

if __name__ == "__main__":
    main()
