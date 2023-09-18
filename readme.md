# Summarize Emails

This project is designed to process and summarize email newsletters and their associated links using the OpenAI API. It fetches emails from a Gmail account, extracts links (if specified in the config file), generates topics, groups and consolidates topics, and sends a summary email.

The method of summarization is custom-designed because the closest "in-built" option I could find was LangChain's MapReduce, which doesn't do a great job. Here we first extract a bunch of topics, then cluster them using KMeans, then summarize the clusters. 

## License

This project is licensed under the AGPL-3.0 License.

## Installation

1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Set up the `.env` file using the provided `.env.example` file. Input your email address and OpenAI API key, and rename the file to `.env`.

## Configuration

The `config.json` file contains settings for email summarization, link crawling, and filtering. Adjust the settings as needed - in particular, you will need to edit the labels in the EMAIL_LIST

## Usage

Run the `main.py` script to start the email summarization process. The script will authenticate with the Gmail API, retrieve emails, generate and consolidate topics, and send a summary email.

## Todo

1. Enable more weight to be placed on specified topics of interest
2. Store source data in vector store enabling user to dive deeper into the summaries as required
3. Add podcasts and other sources to summarise
4. Update to use the new Function Calling functionality here: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
5. Update to use the newest text summarization prompt techniques, eg https://twitter.com/AlphaSignalAI/status/1703825582889263473
6. Launch as a chrome extension?
7. This will still error if given too much data to summarize which might result in a final call of above 16k tokens. I have found that roughly 20 emails including links is a maximum to process (which for me is around 2 weeks or so)
8. ...?