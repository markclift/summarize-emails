# Summarize Emails

This project is designed to process and summarize emails and their associated links using the OpenAI API. It fetches emails from a Gmail account, extracts links, generates topics, consolidates topics, and sends a summary email.

This Readme.md file was generated using OpenAI and the summarize-repo-for-GPT project found here: https://github.com/markclift/summarize-repo-for-GPT.

## License

This project is licensed under the AGPL-3.0 License.

## Installation

1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Set up the `.env` file using the provided `.env.example` file. Input your GitHub API token, email address, and OpenAI API token, and rename the file to `.env`.

## Configuration

The `config.json` file contains settings for email summarization, link crawling, and filtering. Adjust the settings as needed - in particular, you will need to edit the labels in the EMAIL_LIST

## Usage

Run the `main.py` script to start the email summarization process. The script will authenticate with the Gmail API, retrieve emails, generate and consolidate topics, and send a summary email.