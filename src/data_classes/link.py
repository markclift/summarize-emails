from dataclasses import dataclass, field
import hashlib
from scraping.link_processor import extract_content_from_url
import tiktoken

@dataclass
class Link:
    orig_url: str
    final_url: str = field(default=None)
    contents: str = field(init=False)
    contents_with_metadata: str = field(init=False)
    tokens: int = field(init=False)
    link_hash: str = field(init=False)

    def __post_init__(self):
        url_to_use = self.final_url if self.final_url is not None else self.orig_url
        self.final_url, self.contents = extract_content_from_url(url_to_use)
        self.link_hash = hashlib.md5(self.final_url.encode()).hexdigest()[:10]  # Only use the first 10 characters.
        link_metadata = f"Contents of link {self.final_url} referenced in email:\n"
        self.contents_with_metadata = link_metadata + self.contents + '\n\n'
        enc = tiktoken.get_encoding("cl100k_base")
        self.tokens = len(enc.encode(self.contents_with_metadata))