from email_reader import authenticate_gmail, create_email, get_unprocessed_emails, insert_email, mark_email_as_processed
from link_processor import extract_content_from_url, extract_links_from_email
from openai_interface import OpenAI_Interface
from datetime import datetime
import os

def main():
    # Authenticate to Gmail API.
    gmail_service = authenticate_gmail()
    if gmail_service is None:
        print("Failed to authenticate to Gmail API.")
        return

    # Get unprocessed emails.
    days = 7 # For now let's hardcode a 7-day window 
    emails = get_unprocessed_emails(gmail_service, days)

    openai_interface = OpenAI_Interface()
    summaries = []

    # Process each email.
    for count, email in enumerate(emails, 1):
        # Extract links from the email.
        links = extract_links_from_email(email['body'])

        # Fetch and process web content from each link.
        consolidated_content=f"Email {count}:\nSubject: {email['subject']}\nSender: {email['from']}\nDate: {email['date']}\nBody: {email['body']}\n\n"
        for link in links:
            consolidated_content += f"Contents of link {link} referenced in email:\n" + extract_content_from_url(link) + '\n\n'

        # Summarize the content.
        summary = openai_interface.generate_summary(text_to_summarize=consolidated_content, topics_of_interest=["AI", "Decentralised Identity"])

        # Add to the list of summaries.
        summaries.append(summary)

        # Mark email as processed.
        mark_email_as_processed(gmail_service, email['id'])
    
    # Now generate summaries and insert the summary email
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    tokens_total_prompt, tokens_total_completion = openai_interface.get_tokens_total()
    cost_string = f"Tokens summarised: {tokens_total_prompt}, Tokens output: {tokens_total_completion}. Cost of summarization: ${openai_interface.get_token_cost()}"
    email_body = "Summary generated on " + date_time + " covering " + str(len(emails)) + " emails.\n\n" + cost_string + "\n\n".join(summaries)
    message = create_email(os.getenv('MY_EMAIL_ADDRESS'), "Summarize-emails", email_body)
    insert_email(gmail_service, 'me', message)

if __name__ == "__main__":
    main()