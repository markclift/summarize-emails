from goose3 import Goose
from requests import Session
import re
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import time
from config import LINK_TEXT_TO_IGNORE, LINKS_CLASSES_TO_IGNORE, PATHS_TO_IGNORE, UNSUBSCRIBE_KEYWORDS, URLS_TO_IGNORE

from scraping.browser_scraper import scrape_page, scrape_tweet

def find_urls_in_text(email_body):
    # Here we use a simple regex to match URLs.
    # This will only match simple URLs and might not handle all cases.
    urls = re.findall(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        email_body,
    )
    return list(set(urls))

def find_urls_in_alinks(alinks, subject):
    links = []
    valid_alink_texts = []
    for alink in alinks:
        text = alink.text.replace('\n','').strip()
        if (alink.get('href') != '' 
             and text != subject
             and text not in LINK_TEXT_TO_IGNORE
             and alink.get('class') not in LINKS_CLASSES_TO_IGNORE):
            valid_alink_texts.append(text)
            links.append(alink.get('href'))
    print(f'\nFound following alink texts in {subject}: \n"' + '" | "'.join(valid_alink_texts) + '"')
    return links

def clean_url(url):
    if "&utm_source" in url:
        url = url.split("&utm_source")[0]
    elif "?utm_source" in url:
        url = url.split("?utm_source")[0]
    if "?token=" in url:
        url = url.split("?token")[0]
    url = url.replace('%0D%0A','')
    return url


def find_redirect_urls(urls):
    # If the links are redirects, replace them with the final URLs
    final_urls = {}
    print("\nReplacing redirects with actual links")
    for url in tqdm(urls):
        final_urls[url] = find_redirect_url(url)
    return final_urls

def find_redirect_url(url):
    # If the links are redirects, replace them with the final URLs
    session = requests.Session()  # using a Session object
    headers = {"User-Agent": "Mozilla/5.0"}  # setting a user-agent header
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = "https://" + url
    for _ in range(3):  # try 3 times
        try:
            response = session.get(url, headers=headers)
            url = response.url
        except requests.exceptions.RequestException:
            time.sleep(1)  # if failed, wait a bit and then retry
    url = clean_url(url)
    return url

def is_filtered(url):
    return (
        any(domain in url for domain in URLS_TO_IGNORE) or
        not urlparse(url).path or
        '/subscribe/' in url or
        urlparse(url).path in PATHS_TO_IGNORE or
        any(url.endswith(path) for path in PATHS_TO_IGNORE) or
        ("twitter.com" in url and "status" not in url) or
        any(keyword in url for keyword in UNSUBSCRIBE_KEYWORDS)
    )

def is_twitter_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc == "twitter.com"


def extract_content_from_url(url):
    final_url=url
    url_content=''
    if is_twitter_url(url):
        final_url, url_content = scrape_tweet(url)
    elif url.startswith("mailto:"):
        pass
    else:
        session = Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        g = Goose({'browser_user_agent': 'Mozilla', 'http_session': session})
        try:
            extract = g.extract(url=url)
            if extract.cleaned_text.startswith('JavaScript is not available'):
                if extract.links[0].startswith('https://help.twitter.com'):
                    final_url, url_content = scrape_tweet(url)
                else:
                    final_url, url_content = scrape_page(url)
            else: 
                final_url=extract.canonical_link
                url_content=extract.cleaned_text
        except Exception as e: # Catch all exceptions
            print(f"An error occurred while extracting content from {url}\n{e}")
        
        #I don't know why but sometimes the canonical link is still a redirect
        if 'redirect' in final_url:
            final_url = find_redirect_url(url)
        final_url = clean_url(final_url)
    
    return final_url, url_content