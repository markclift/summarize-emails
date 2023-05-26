from goose3 import Goose
import re

def extract_links_from_email(email_body):
    # Here we use a simple regex to match URLs.
    # This will only match simple URLs and might not handle all cases.
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_body)
    return urls

def extract_content_from_url(url):
    g = Goose()
    return g.extract(url=url).cleaned_text