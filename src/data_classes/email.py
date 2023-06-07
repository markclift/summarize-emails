from dataclasses import dataclass, field
from typing import List
import tiktoken
from data_classes.link import Link

@dataclass
class Email:
    id: str
    body: str
    urls_list: List[str]
    from_value: str
    subject_value: str
    date_value: str
    Links_list: List[Link] = field(init=False)
    metadata: str = field(init=False)
    body_with_metadata: str = field(init=False)

    def __post_init__(self):
        self.metadata = f"Email {self.id}:\nSubject: {self.subject_value}\nSender: {self.from_value}\nDate: {self.date_value}\n"
        self.body_with_metadata = self.metadata + f"Body: {self.body}\n\n"
        enc = tiktoken.get_encoding("cl100k_base")
        self.tokens = len(enc.encode(self.body_with_metadata))
        self.Links_list = []

    def update_link(self, orig_link, final_link):
        self.body.replace(orig_link, final_link)
        self.body_with_metadata.replace(orig_link, final_link)
        return self