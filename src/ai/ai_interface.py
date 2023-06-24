import os
import re
from typing import List
import openai
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
from scipy.spatial.distance import cosine

import tiktoken
from config import (
    CONSOLIDATE_TITLES_MODEL,
    SUMMARIZE_SUMMARIES_MODEL,
    EMBEDDINGS_MODEL,
)
from data_classes.llm import LLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from utils import split_text

# Removed bc wasn't useful - Here are some example categories but you can also choose others: [Defi, NFT, AI, Funding, Decentralized Identity, Gaming, Art, Governance, DAOs, Staking, Market Trends, Zero Knowledge, Stablecoins, Bridges].

NOT_FOUND_TXT = "NO TOPICS FOUND"
TOPICS_SYSTEM_PROMPT = (
    "You respond with either '"
    + NOT_FOUND_TXT
    + """' or a list of mutually exclusive and collectively exhaustive topics. Topics should ignore any advertisements or sponsored articles or encouragements to subscribe or engage with the author and should be more specific than general topics like 'News', 'News Highlights', 'Recent News', 'Various News' or 'Trending topics'. If there is funding news, please list the funding. Each topic should have a title describing the topic and a summary. Neither the title nor the description should contain new line characters. The summaries should be a minimum of 80 words. Return your answer in the following format - a numbered list, with a new line separating each topic like these examples: 
        1. Example Topic Title 1 | This is a summary of the topic'
        2. Example Topic Title 2 | This is a summary of the topic'
        3. Example Topic Title 3 | This is a summary of the topic'"""
)
TOPICS_USER_PROMPT = (
    """User: Give me a list of {num_topics} mutually exclusive and collectively exhaustive topics that together summarize the following text. If the text is empty or nearly empty or not meaningful, just return '"""
    + NOT_FOUND_TXT
    + """'. Here is the text:\n\n{text_to_summarize}\n\n\nTOPICS:"""
)
TITLE_SYSTEM_PROMPT = """You are a wise being able to summarize text succicently. You Return your answer in a numbered list, with a new line separating each title as in these examples: 
        1. ❄️ Title 1
        2. 🏆 Title 2
        3. 🚀 Title 3"""
TITLE_USER_PROMPT = """Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible, 
        and do not overlap with each other. Add the most relevant emoji to the start of each title:
        {text}

        TITLES:
        """
FINAL_SUMMARIES_SYSTEM_PROMPT = "You are a wise being, able to summarize text clearly."
FINAL_SUMMARIES_USER_PROMPT = """Write a 500-word summary of the following, removing duplicate information:
{text}
    
    500-WORD SUMMARY:"""


class AI_Interface:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.models = []

    def get_model(self, model_name: str):
        for model in self.models:
            if model.model_name == model_name:
                return model
        new_model = LLM(model_name)
        self.models.append(new_model)
        return new_model

    def get_total_cost(self):
        cost = 0
        for model in self.models:
            cost += model.get_cost()
        return cost

    def parse_topics(self, text):
        topics = text.split("\n")  # splitting the text into topics by '\n\n'
        topic_dicts = []

        for topic in topics:
            if (
                topic != ""
                and topic.strip().replace(".", "") != NOT_FOUND_TXT
                and not topic.lower()
                .strip()
                .replace(".", "")
                .endswith(NOT_FOUND_TXT.lower())
                and topic.lower().strip().replace(".", "") != "empty string"
                and not topic.startswith("This is an example topic")
            ):
                # splitting each topic into title and summary by ' | '
                topic = topic.replace("\n", "")
                try:
                    title, summary = topic.split(" | ", 1)
                except Exception as error:
                    print(
                        "\n!!!Malformed response from GPT - skipping: " + topic + "\n"
                    )
                title = re.sub(
                    "^\d+\.\s*", "", title
                ).strip()  # Removes a number and a dot if the API returns the topics as a numbered list
                topic_dict = {"topic_title": title, "topic_summary": summary}
                topic_dicts.append(topic_dict)

        return topic_dicts

    def _generate_topics_and_summaries(self, chunks: List[str], model, num_topics=10):
        topics = []
        for chunk in chunks:
            chunk = re.sub("\n{3,}", "\n\n", chunk).strip("\n")
            print(f"Looking for {num_topics} topics in: \n" + chunk + "\n")
            user_prompt = TOPICS_USER_PROMPT.replace(
                "{text_to_summarize}", chunk
            ).replace("{num_topics}", str(num_topics))
            topics.extend(
                self.parse_topics(
                    self._generate_chat_completion(
                        model,
                        TOPICS_SYSTEM_PROMPT,
                        user_prompt,
                        model.config["max_tokens_response"],
                    )
                )
            )
        return topics

    def combine_and_generate_topics_and_summaries(
        self, texts: List[str], model_name: str
    ):
        model = self.get_model(model_name)
        enc = tiktoken.encoding_for_model(model.model_name)
        consolidated_texts = []
        current_text = ""
        current_token_count = 0

        for text in texts:
            tokens = len(enc.encode(text))
            if current_token_count + tokens <= model.config["max_tokens_input"]:
                # If adding this text won't exceed the limit, add it to current text
                current_text += "\n\n" + text
                current_token_count += tokens
            else:
                # If adding this text would exceed the limit, start a new text
                consolidated_texts.append(current_text)
                current_text = text
                current_token_count = tokens

        # Append the last text if it's not empty
        if current_text:
            consolidated_texts.append(current_text)

        return self._generate_topics_and_summaries(consolidated_texts, model)

    def split_and_generate_topics_and_summaries(
        self, text: str, model_name: str, backup_model_name="", num_topics=10
    ):
        model = self.get_model(model_name)
        enc = tiktoken.encoding_for_model(model_name)
        tokens = len(enc.encode(text))
        chunks = []
        if tokens > model.config["max_tokens_input"]:
            if backup_model_name != "":
                model = self.get_model(backup_model_name)
                if tokens < model.config["max_tokens_input"]:
                    chunks.append(text)
                else:
                    print(f"Text is too long ({str(tokens)} tokens). Chunking...")
                    chunks = split_text(text, tokens, model.config["max_tokens_input"])
            else:
                print(f"Text is too long ({str(tokens)} tokens). Chunking...")
                chunks = split_text(text, tokens, model.config["max_tokens_input"])
        else:
            chunks.append(text)
        return self._generate_topics_and_summaries(chunks, model, num_topics)

    @retry(wait=wait_random_exponential(min=10, max=120), stop=stop_after_attempt(10))
    def completion_with_backoff(self, model, messages, max_tokens, temperature):
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _generate_chat_completion(
        self, model, system_prompt, user_prompt, max_tokens=2000
    ):
        start = datetime.now()
        print(f"\ngenerate_chat_completion start time: {start}")

        gpt_message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            response = self.completion_with_backoff(
                model=model.model_name,
                messages=gpt_message,
                max_tokens=max_tokens,
                temperature=0,
            )

            # extract the response content
            print(response)
            response_content = response["choices"][0]["message"]["content"].strip()
            model.prompt_tokens += response["usage"]["prompt_tokens"]
            model.completion_tokens += response["usage"]["completion_tokens"]
            cost = model.get_cost(
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )
            print(f"Cost (USD): ${cost}")

            return response_content

        except Exception as error:
            raise error
        finally:
            end = datetime.now()
            duration = int((end - start).total_seconds())
            print(
                f"generate_chat_completion completed time {end}\nDuration: {duration} seconds"
            )

    def group_topics(self, topics):
        # Create embeddings for all topics
        responses = openai.Embedding.create(
            input=[topic["topic_summary"] for topic in topics], model=EMBEDDINGS_MODEL
        )

        # Extract the embeddings from the responses
        embeddings = [response["embedding"] for response in responses["data"]]

        # Create a matrix from the embeddings
        matrix = np.vstack(embeddings)

        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        kmeans.fit(matrix)

        labels = kmeans.labels_

        # Create a list of tuples, where each tuple is a topic and its corresponding cluster
        topics_with_clusters = list(zip(topics, labels))

        # Create a dictionary where keys are cluster labels and values are lists of topics in the cluster
        suggested_topic_groupings = {}
        for i, (topic, label) in enumerate(topics_with_clusters):
            if label not in suggested_topic_groupings:
                suggested_topic_groupings[label] = []
            suggested_topic_groupings[label].append(
                {
                    "summary_id": i,
                    "topic_summary": topic["topic_summary"],
                    "topic_title": topic["topic_title"],
                }
            )

        return suggested_topic_groupings

    def summarize_titles(self, topics_titles_concat_all):
        output = self._generate_chat_completion(self.get_model(CONSOLIDATE_TITLES_MODEL), TITLE_SYSTEM_PROMPT, TITLE_USER_PROMPT.replace("{text}", topics_titles_concat_all), 1000)

        # Extract the message content from the chat output
        titles = output.split("\n")

        # Remove any empty titles
        titles = [t.strip() for t in titles if t.strip() != ""]
        return titles
    
    def summarize_groups(self, suggested_topic_groupings, summary_num_words=500):
        start = datetime.now()
        print(f"summarize_groups start time {start}")

        topics_data = []
        for i, cluster in suggested_topic_groupings.items():
            topic_data = {}
            topic_data["summaries_concat"] = " ".join(
                item["topic_summary"] for item in cluster
            )
            topic_data["titles_concat"] = ", ".join(
                item["topic_title"] for item in cluster
            )
            topics_data.append(topic_data)

        # Get a list of each community's summaries (concatenated)
        topics_summary_concat = [c["summaries_concat"] for c in topics_data]
        topics_titles_concat = [c["titles_concat"] for c in topics_data]
        print("Suggested topic groupings: " + str(topics_titles_concat))

        # Concat into one long string to do the topic title creation
        topics_titles_concat_all = """"""
        for i, c in enumerate(topics_titles_concat):
            topics_titles_concat_all += f"""{i+1}. {c}
            """
        titles = self.summarize_titles(topics_titles_concat_all)

        # Concat into one long string to do the topic title creation
        summaries=[]
        for summary in topics_summary_concat:
            summaries.append(self._generate_chat_completion(self.get_model(SUMMARIZE_SUMMARIES_MODEL), FINAL_SUMMARIES_SYSTEM_PROMPT, FINAL_SUMMARIES_USER_PROMPT.replace("{text}", summary), 1200))

        final_outputs = [
            {"topic_title": t, "topic_summary": s.replace("\n", "").strip()}
            for t, s in zip(titles, summaries)
        ]

        end = datetime.now()
        duration = int((end - start).total_seconds())
        print(f"summarize_groups completed time {end}\nDuration: {duration} seconds")

        return final_outputs