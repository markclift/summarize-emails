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
            creds.refresh(Request())
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

def get_unprocessed_emails(service, days):
    """Fetch all emails with "3 News/Crypto News" label and without "Processed" label."""
    start_date = datetime.now() - timedelta(days=days)
    start_date_formatted = start_date.strftime("%Y/%m/%d")
    try:
        # Retrieve label IDs for "Newsletter" and "Processed".
        label_crypto_news_name = "3-news-crypto-news"
        label_crypto_news_id = get_label_id(service, "3 News/Crypto News")
        label_processed_name = "Processed"
        label_processed_id = get_label_id(service, label_processed_name)

        # Use the Gmail API to fetch these emails.
        response = {'nextPageToken': None}
        email_ids = []
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            q = f"label:{label_crypto_news_name} -label:{label_processed_name} after:{start_date_formatted}"
            response = service.users().messages().list(userId='me', q=q, pageToken=page_token).execute()
            if 'messages' in response:
                email_ids.extend(response['messages'])

        emails=[]
        for id in email_ids:
            response = service.users().messages().get(userId='me', id=id['id']).execute()
            if 'payload' in response:
                body = response['payload']['parts'][0]['body']['data']
                decoded_body = base64.urlsafe_b64decode(body).decode('utf-8')
                headers = response['payload']['headers']
                from_value = next((header['value'] for header in headers if header['name'] == 'From'), None)
                date_value = next((header['value'] for header in headers if header['name'] == 'Date'), None)
                subject_value = next((header['value'] for header in headers if header['name'] == 'Subject'), None)
                emails.append({
                    'id': id['id'],
                    'body': decoded_body,
                    'from': from_value,
                    'subject': subject_value,
                    'date': date_value
                })
        
        return emails

    except (HttpError, RefreshError) as error:
        print(f'An error occurred: {error}')
        return None


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

def create_email(to, subject, body):
    """Create a raw RFC 2822 formatted message."""
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    return {
        'raw': raw_message
    }

def insert_email(service, user_id, message):
    """Insert an email message."""
    try:
        message = (service.users().messages().insert(userId=user_id, body=message)
                   .execute())
        print('Message Id: %s' % message['id'])
        return message
    except Exception as error:
        print(f'An error occurred: {error}')
        return None