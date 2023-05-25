from email_reader import authenticate_gmail, create_email, get_unprocessed_emails, insert_email, mark_email_as_processed
from link_processor import extract_links_from_email, fetch_web_content, parse_web_content
from summarizer import summarize_text
from datetime import datetime
import os

def main():
    """Main function to run the email summarization process."""
    # Authenticate to Gmail API.
    gmail_service = authenticate_gmail()
    if gmail_service is None:
        print("Failed to authenticate to Gmail API.")
        return

    # Get unprocessed emails.
    emails = get_unprocessed_emails(gmail_service)

    summaries = []

    # Process each email.
    for email in emails:
        email_body = email['body']  # Replace with real email body.
        email_id = email['id']  # Replace with real email ID.

        # Extract links from the email.
        links = extract_links_from_email(email_body)

        # Fetch and process web content from each link.
        for link in links:
            html_content = fetch_web_content(link)
            if html_content is not None:
                text_content = parse_web_content(html_content)

                # Summarize the text content.
                summary = summarize_text(text_content, topics=["AI", "Decentralised Identity"])

                # Add to the list of summaries.
                summaries.append(summary)

        # Mark email as processed.
        mark_email_as_processed(gmail_service, email_id)
    
    # Here, you'd compose and send the summary email.
    # We won't cover that in this code, but you'd use the Gmail API similarly to how we did above.
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    email_body = "Summary generated on " + date_time + " covering " + str(len(emails)) + " emails.\n\n" + "\n".join(summaries)
    message = create_email(os.getenv('MY_EMAIL_ADDRESS'), "Summarize-emails", email_body)
    insert_email(gmail_service, 'me', message)

if __name__ == "__main__":
    main()