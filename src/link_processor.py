import requests
from bs4 import BeautifulSoup
import re

def extract_links_from_email(email_body):
    """Extract all URLs from an email's body."""
    # Here we use a simple regex to match URLs.
    # This will only match simple URLs and might not handle all cases.
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
    return urls

def fetch_web_content(url):
    """Fetch the HTML content of a webpage given a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    return None

def parse_web_content(html_content):
    """Parse HTML content and extract meaningful text."""
    # This function uses BeautifulSoup to extract the meaningful text from the HTML content.
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Removes all script and style elements
    for script in soup(['script', 'style']):
        script.decompose()

    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text