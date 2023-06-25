import math
from config import EMAIL_LIST, SUMMARIZE_EMAIL_MODEL, SUMMARIZE_LINKS_MODEL, SUMMARIZE_LINKS_BACKUP_MODEL
from scraping.email_interface import authenticate_gmail, create_email, get_cleaned_emails_and_links, mark_email_as_processed, send_email
from scraping.link_processor import find_redirect_urls, is_filtered
from ai.ai_interface import AI_Interface
import os
from data_classes.link import Link
import logging

logging.basicConfig(level=logging.INFO)

DATA_FOLDER='data/'
TOPICS_OF_INTEREST=["AI", "Decentralized Identity"] #TODO 
WINDOW_DAYS=10
MAX_EMAILS = -1 #TODO: Use for testing. Set to -1 in prod
MAX_SUMMARIES = -1 #TODO: Use for testing. Set to -1 in prod

def get_suggested_topics(num_tokens):
    if num_tokens < 400:
        return 1
    else:
        return min(math.ceil((num_tokens - 400 + 1) / 200) + 1,10)

def download_link_contents(emails_with_links):
    for count, email_object in enumerate(emails_with_links, 1):
        logging.info(f"\nProcessing email {count} of {len(emails_with_links)}: "+ email_object.metadata)
        logging.info("Email tokens: "+ str(email_object.tokens))
        
        final_urls = find_redirect_urls(email_object.urls_list)
        for orig_url, final_url in final_urls.items():
            email_object = email_object.update_link(orig_url, final_url)
            if not is_filtered(final_url) and not any(obj.final_url == final_url for obj in email_object.Links_list):
                link_object = Link(orig_url=orig_url, final_url=final_url)
                if link_object.contents == '':
                    logging.warning("Couldn't download this link contents: " + link_object.final_url)
                else:
                    logging.info("Downloaded link contents: " + link_object.final_url)
                    email_object.Links_list.append(link_object)
                    logging.info("Link tokens: " + str(link_object.tokens))

def generate_topics(emails_with_links, crawl_links, ai_interface):
    topics=[]
    for count, email_object in enumerate(emails_with_links, 1):
        logging.info(f"\nSummarising email {count} of {len(emails_with_links)}: "+ email_object.metadata)
        logging.info("Email tokens: "+ str(email_object.tokens))

        topics.extend(ai_interface.split_and_generate_topics_and_summaries(email_object.body_with_metadata, SUMMARIZE_EMAIL_MODEL))

        if crawl_links == True:
            for count_link, link in enumerate(email_object.Links_list, 1):
                if count_link == MAX_SUMMARIES-1: break #For testing so we don't have to wait forever
                if link.tokens > 100:
                    num_topics = get_suggested_topics(link.tokens)
                    logging.info(f"\nSummarising link {count_link} of {len(email_object.Links_list)} (in email {count} of {len(emails_with_links)}): "+ link.final_url)
                    logging.info("Link tokens: "+ str(link.tokens))
                    topics.extend(ai_interface.split_and_generate_topics_and_summaries(link.contents, SUMMARIZE_LINKS_MODEL, SUMMARIZE_LINKS_BACKUP_MODEL, num_topics))
                else: logging.info(f"\Skipping link {count_link} of {len(email_object.Links_list)} (in email {count} of {len(emails_with_links)}) because too short ({str(link.tokens)} tokens)")
    return topics

def consolidate_topics(topics, ai_interface):
    logging.info('\nTotal model cost so far: $' + str(round(ai_interface.get_total_cost(),2)))
    logging.info('\n\n==============================================================\n\n')

    suggested_topic_groupings = ai_interface.group_topics(topics)
    topics_final = ai_interface.summarize_groups(suggested_topic_groupings)

    logging.info('\n\n==============================================================\n')
    for count, topic in enumerate(topics_final,1):
        logging.info(f"\nTopic {count} of {len(topics_final)}: " + topic['topic_title'] + '\n' + topic['topic_summary'])
    logging.info('\nTotal model cost: $' + str(round(ai_interface.get_total_cost(),2)))
    
    return topics_final

def main():
    # Setup
    ai_interface = AI_Interface()

    # Authenticate to Gmail API.
    gmail_service = authenticate_gmail()
    if gmail_service is None:
        logging.error("Failed to authenticate to Gmail API.")
        return
    
    for email_to_create in EMAIL_LIST:
        logging.info(f"CREATING: {email_to_create['subject']} using these email labels: {email_to_create['labels']}")

        # Retrieve list of emails from Gmail
        emails_with_links = get_cleaned_emails_and_links(gmail_service, WINDOW_DAYS, email_to_create['labels'], MAX_EMAILS)
        if email_to_create['crawl_links'] == True:
            download_link_contents(emails_with_links)

        if len(emails_with_links) > 0:
            # Generate topics
            topics_initial = generate_topics(emails_with_links, email_to_create['crawl_links'], ai_interface)

            # Consolidate topics
            topics_final = consolidate_topics(topics_initial, ai_interface)

            # Insert the summary email
            message = create_email(os.getenv('MY_EMAIL_ADDRESS'), email_to_create['subject'], topics_final, ai_interface.get_total_cost(), emails_with_links)
            send_email(gmail_service, 'me', message)

            # Mark emails as processed.
            for email in emails_with_links:
                mark_email_as_processed(gmail_service, email.id)
        
        else:
            logging.info(f"No unprocessed emails found with labels: {email_to_create['labels']}")

if __name__ == "__main__":
    main()