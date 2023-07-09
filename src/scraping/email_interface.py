import re
from typing import Any, Dict
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError
import os
import pickle
from email.mime.text import MIMEText
import base64
from datetime import datetime, timedelta

from scraping.link_processor import find_urls_in_alinks, find_urls_in_text
from data_classes.email import Email

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def authenticate_gmail():
    """Authenticate to Gmail API."""
    creds = None

    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                os.remove('token.pickle')
                return authenticate_gmail()
        else:
            flow = InstalledAppFlow.from_client_secrets_file('./gmail-credentials.json', SCOPES)
            creds = flow.run_local_server(port=58599)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def clean_email(email_body):
    # First, remove the line that starts with 'View this post on the web at' so we don't duplicate the email body - this is common in Substack
    email_body = re.sub(r"View this post on the web at .*", "", email_body)
    # Then remove the unsubscribe URL that comes directly after the text 'Unsubscribe '
    email_body = re.sub(
        r"Unsubscribe http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        email_body,
    )
    email_body = email_body.replace('\xa0', '').replace('\u200c', '')
    return email_body

def get_label_id(service, label_name):
    """Retrieve the ID of a Gmail label given its name."""
    try:
        result = service.users().labels().list(userId='me').execute()
        labels = result.get('labels', [])

        for label in labels:
            if label['name'].lower() == label_name.lower():
                return label['id']
    
    except (HttpError, RefreshError) as error:
        print(f'An error occurred: {error}')

    return None

def get_cleaned_emails_and_links(service, days, label_list, max_emails):
    """Fetch all emails with designated labels and without "Processed" label."""
    start_date = datetime.now() - timedelta(days=days)
    start_date_formatted = start_date.strftime("%Y/%m/%d")
    label_string = ' OR '.join([f"label:{label} " for label in label_list])
    try:
        response, email_ids = {'nextPageToken': None}, []
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            q = label_string+f"-label:Processed after:{start_date_formatted}"
            response = service.users().messages().list(userId='me', q=q, pageToken=page_token).execute()
            email_ids.extend(response.get('messages', []))

        emails=[]
        for count, id in enumerate(email_ids, 1):
            emails.append(_process_email(service, id))
            if count==max_emails: break
        return emails
    except (HttpError, RefreshError) as error:
        print(f'An error occurred: {error}')
        return None

def _process_email(service, id: Dict[str, str]) -> Dict[str, Any]:
    response = service.users().messages().get(userId='me', id=id['id']).execute()
    payload = response.get('payload', {})
    headers = payload.get('headers', [])
    subject_value = next((header['value'] for header in headers if header['name'] == 'Subject'), None)
    from_value = next((header['value'] for header in headers if header['name'] == 'From'), None)
    date_value = next((header['value'] for header in headers if header['name'] == 'Date'), None)
    body, links = get_email_body_and_links(payload, subject_value)
    return Email(
        id=id['id'],
        body=body,
        urls_list=links,
        from_value=from_value,
        subject_value=subject_value,
        date_value=date_value
    )

def get_email_body_and_links(payload, subject):
    data = ''
    links=[]
    if 'parts' in payload:
        parts = payload['parts']
        html_part = None
        text_part = None
        for part in parts:
            if part['mimeType'] == "text/html":
                html_part = part
            elif part['mimeType'] == "text/plain":
                text_part = part
        if html_part:
            data, links = _get_email_body_and_links(html_part, subject)
        elif text_part:
            data, links = _get_email_body_and_links(text_part, subject)
    else:
        data, links = _get_email_body_and_links(payload, subject)
    return data, links
        
def _get_email_body_and_links(part_or_payload, subject):
    mimeType = part_or_payload['mimeType']
    data = part_or_payload['body']['data']
    if mimeType == "text/plain":
        email_body = clean_email(base64.urlsafe_b64decode(data).decode())
        links = find_urls_in_text(email_body)
    elif mimeType == "text/html":
        html = base64.urlsafe_b64decode(data).decode()
        soup = BeautifulSoup(html, "html.parser")
        #Let's keep the paragraph structure for now. Only way to do that is translate html to '\n' before using BeautifulSoup's get_text otherwise they will be lost
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for p in soup.find_all("p"):
            p.append("\n")
        email_body = clean_email(soup.get_text())
        alinks = soup.find_all('a')
        links = find_urls_in_alinks(alinks, subject)
    return email_body, links

def mark_email_as_processed(service, email_id):
    """Mark an email as read, apply the "Processed" label, and archive it."""
    try:
        # Retrieve label ID for "Processed".
        processed_label_id = get_label_id(service, "Processed")

        # Use the Gmail API to modify the email's labels.
        message = service.users().messages().modify(userId='me', id=email_id,
            body={'removeLabelIds': ['UNREAD', 'INBOX'], 'addLabelIds': [processed_label_id]}).execute()

        print(f"Email {message['id']} marked as processed.")
    
    except (HttpError, RefreshError) as error:
        print(f'An error occurred: {error}')

def create_email(to, subject, topics_final, cost, email_list):
    
    cost = "{:.2f}".format(cost)
    email_body = f"<p>The following emails were summarised with a cost of ${cost}:</p><ul>"
    for email in email_list:
        # use regex to remove the timezone abbreviation
        email.date_value = re.sub(r'\s\(\w+\)', '', email.date_value).strip()
        date_obj = datetime.strptime(email.date_value, "%a, %d %b %Y %H:%M:%S %z")
        formatted_date = date_obj.strftime("%a, %d %b %Y %H:%M")
        email_body += f"<li style='list-style-type: none;'><span style='margin-right: 10px;'>✅</span>{email.subject_value} | {email.from_value} | {formatted_date}\n</li>"
    email_body += "</ul>"
    email_body += f"<h2>Summary:</h2>"
    # Append topics_final to email_body
    for count, topic in enumerate(topics_final, 1):
        clean_title = re.sub('^\d+\.\s', '', topic['topic_title'])
        email_body += f"<p><b><u>Topic {count} of {len(topics_final)}: {clean_title}</u></b></p>"
        email_body += f"<p>{topic['topic_summary']}</p>"
    
    """Create a raw RFC 2822 formatted message."""
    message = MIMEText(email_body, 'html')  # Here 'html' is specified to indicate the body content is in HTML
    message['from'] = to
    message['to'] = to
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    return {
        'raw': raw_message
    }

def insert_email(service, user_id, message):
    """Insert an email message."""
    try:
        message['labelIds'] = ['INBOX', 'UNREAD']
        message = (service.users().messages().insert(userId=user_id, body=message)
                   .execute())
        return message
    except Exception as error:
        print(f'An error occurred: {error}')
        return None
    
def send_email(service, user_id, message):
    """Send an email message."""
    try:
        message['labelIds'] = ['INBOX', 'UNREAD']
        message = (service.users().messages().send(userId=user_id, body=message)
                   .execute())
        return message
    except Exception as error:
        print(f'An error occurred: {error}')
        return None
