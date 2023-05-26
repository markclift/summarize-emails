from goose3 import Goose
import re
import requests
from urllib.parse import urlparse

from twitter_scraper import scrape_tweet

URLS_TO_IGNORE=['milkroad.com', 'dune.com', 'coingecko.com']

def extract_links_from_email(email_body):
    # Here we use a simple regex to match URLs.
    # This will only match simple URLs and might not handle all cases.

    # First, remove the line that starts with 'View this post on the web at' so we don't duplicate the email body - this is common in Substack
    email_body = re.sub(r'View this post on the web at .*', '', email_body)
    # Then remove the unsubscribe URL that comes directly after the text 'Unsubscribe '
    email_body = re.sub(r'Unsubscribe http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'Unsubscribe ', email_body)

    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)

    #If the links are redirects, replace them with the final URLs
    final_urls = []
    for url in urls:
        response = requests.get(url)
        final_urls.append(response.url)

    #Clean the URLs
    clean_urls = [url.split('?utm_source=')[0] for url in final_urls]

    # Finally, filter out URLs that contain 'twitter.com'
    final_urls_filtered = [url for url in clean_urls if not any(domain in url for domain in URLS_TO_IGNORE) and urlparse(url).path != '/']

    #Remove duplicates
    unique_urls = list(set(final_urls_filtered))

    return unique_urls

def is_twitter_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc == 'twitter.com'

def extract_content_from_url(url):
    if is_twitter_url(url):
        return scrape_tweet(url)
    else:
        g = Goose()
        extract = g.extract(url=url)
        return extract.cleaned_text