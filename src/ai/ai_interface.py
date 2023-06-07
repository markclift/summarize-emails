import os
import re
from typing import List
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

import tiktoken
from config import CONSOLIDATE_TITLES_MODEL, MAP_REDUCE_TOPICS_MODEL, SUMMARIZE_LINKS_MODEL
from data_classes.llm import LLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import networkx as nx
from networkx.algorithms import community

from utils import split_text

#Removed bc wasn't useful - Here are some example categories but you can also choose others: [Defi, NFT, AI, Funding, Decentralized Identity, Gaming, Art, Governance, DAOs, Staking, Market Trends, Zero Knowledge, Stablecoins, Bridges]. 

NOT_FOUND_TXT = "NO TOPICS FOUND"
SYSTEM_PROMPT = (
    "You respond with either '"
    + NOT_FOUND_TXT
    + """' or a list of mutually exclusive and collectively exhaustive topics. Topics should ignore any advertisements or sponsored articles or encouragements to subscribe or engage with the author and should be more specific than general topics like 'News', 'News Highlights', 'Recent News', 'Various News' or 'Trending topics'. If there is funding news, please list the funding. Each topic should have a title describing the topic and a summary. Neither the title nor the description should contain new line characters. The summaries should be a minimum of 80 words. Return your answer in the following format - a numbered list, with a new line separating each topic like these examples: 
        1. Example Topic Title 1 | This is a summary of the topic'
        2. Example Topic Title 2 | This is a summary of the topic'
        3. Example Topic Title 3 | This is a summary of the topic'"""
)
USER_PROMPT_TOPIC_TITLES_PLUS_SUMMARIES = (
    """User: Give me a list of {num_topics} mutually exclusive and collectively exhaustive topics that together summarize the following text. If the text is empty or nearly empty or not meaningful, just return '"""
    + NOT_FOUND_TXT
    + """'. Here is the text:\n\n{text_to_summarize}\n\n\nTOPICS:"""
)

class AI_Interface:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.models = []
        self.langchain_llm = OpenAI(temperature=0, model_name=SUMMARIZE_LINKS_MODEL)

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
                and not topic.lower().strip().replace(".", "").endswith(NOT_FOUND_TXT.lower())
                and topic.lower().strip().replace(".", "") != "empty string"
                and not topic.startswith("This is an example topic")
            ):
                # splitting each topic into title and summary by ' | '
                topic = topic.replace("\n", "")
                try:
                    title, summary = topic.split(" | ", 1)
                except Exception as error:
                    print("\n!!!Malformed response from GPT - skipping: " + topic + '\n')
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
            user_prompt = USER_PROMPT_TOPIC_TITLES_PLUS_SUMMARIES.replace(
                "{text_to_summarize}", chunk
            ).replace(
                "{num_topics}", str(num_topics)
            )
            topics.extend(
                self.parse_topics(
                    self._generate_chat_completion(
                        model,
                        SYSTEM_PROMPT,
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

    def split_and_generate_topics_and_summaries(self, text: str, model_name: str, num_topics=10):
        model = self.get_model(model_name)
        enc = tiktoken.encoding_for_model(model_name)
        tokens = len(enc.encode(text))
        chunks = []
        if tokens > model.config["max_tokens_input"]:
            print(f"Text is too long ({str(tokens)} tokens). Chunking...")
            chunks = split_text(text, tokens, model.config['max_tokens_input'])
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

    def gpt3_bulk_generate_topics(self, text_array):
        start = datetime.now()
        print(f"gpt3_bulk_generate_topics start time: {start}")

        map_prompt = PromptTemplate(
            template=USER_PROMPT_TOPIC_TITLES_PLUS_SUMMARIES,
            input_variables=["text_to_summarize"],
        )

        map_llm_chain = LLMChain(llm=self.langchain_llm, prompt=map_prompt)
        map_llm_chain_input = [{"text_to_summarize": t} for t in text_array if t != ""]

        with get_openai_callback() as cb:
            map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)

        topics = []
        for e in map_llm_chain_results:
            topics.extend(self.parse_topics(e["text"]))
        self.gpt_3_model.prompt_tokens += cb.prompt_tokens
        self.gpt_3_model.completion_tokens += cb.completion_tokens
        print(f"\nPrompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${self.gpt_3_model.get_cost()}")

        end = datetime.now()
        duration = int((end - start).total_seconds())
        print(
            f"gpt3_bulk_generate_topics completed time {end}\nDuration: {duration} seconds"
        )

        return topics

    # Run the community detection algorithm
    def _group_topics(
        self, title_similarity, num_topics=8, bonus_constant=0.25, min_size=3
    ):
        proximity_bonus_arr = np.zeros_like(title_similarity)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 0
                else:
                    proximity_bonus_arr[row, col] = (
                        1 / (abs(row - col)) * bonus_constant
                    )

        title_similarity += proximity_bonus_arr

        title_nx_graph = nx.from_numpy_array(title_similarity)

        desired_num_topics = num_topics
        # Store the accepted partitionings
        topics_title_accepted = []

        resolution = 0.85
        resolution_step = 0.01
        iterations = 40

        # Find the resolution that gives the desired number of topics
        topics_title = []
        while len(topics_title) not in [
            desired_num_topics,
            desired_num_topics + 1,
            desired_num_topics + 2,
        ]:
            topics_title = community.louvain_communities(
                title_nx_graph, weight="weight", resolution=resolution
            )
            resolution += resolution_step
        topic_sizes = [len(c) for c in topics_title]
        sizes_sd = np.std(topic_sizes)
        modularity = community.modularity(
            title_nx_graph, topics_title, weight="weight", resolution=resolution
        )

        lowest_sd_iteration = 0
        # Set lowest sd to inf
        lowest_sd = float("inf")

        for i in range(iterations):
            topics_title = community.louvain_communities(
                title_nx_graph, weight="weight", resolution=resolution
            )
            modularity = community.modularity(
                title_nx_graph, topics_title, weight="weight", resolution=resolution
            )

            # Check SD
            topic_sizes = [len(c) for c in topics_title]
            sizes_sd = np.std(topic_sizes)

            topics_title_accepted.append(topics_title)

            if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
                lowest_sd_iteration = i
                lowest_sd = sizes_sd

        # Set the chosen partitioning to be the one with highest modularity
        topics_title = topics_title_accepted[lowest_sd_iteration]
        print(f"Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}")

        topic_id_means = [sum(e) / len(e) for e in topics_title]
        # Arrange title_topics in order of topic_id_means
        topics_title = [
            list(c)
            for _, c in sorted(
                zip(topic_id_means, topics_title), key=lambda pair: pair[0]
            )
        ]
        # Create an array denoting which topic each chunk belongs to
        chunk_topics = [None] * title_similarity.shape[0]
        for i, c in enumerate(topics_title):
            for j in c:
                chunk_topics[j] = i

        return {"chunk_topics": chunk_topics, "topics": topics_title}

    def group_topics(self, topics):
        # Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
        openai_embed = OpenAIEmbeddings()

        summary_embeds = np.array(openai_embed.embed_documents([topic['topic_summary'] for topic in topics]))
        
        # Tried with a cpl of different variations but it didn't work as well as just using the summaries
        #hashtag_embeds = np.array(openai_embed.embed_documents([topic['topic_hashtags'] for topic in topics]))
        #summary_and_hashtag_embeds = np.array(openai_embed.embed_documents([topic['topic_hashtags'] + ' ' + topic['topic_summary'] for topic in topics]))
        #title_embeds = np.array(openai_embed.embed_documents([topic['topic_title'] for topic in topics]))
        num_topics = len(summary_embeds)
        # Get similarity matrix between the embeddings of the chunk summaries
        summary_similarity_matrix = np.zeros((num_topics, num_topics))
        summary_similarity_matrix[:] = np.nan
        
        #hashtag_similarity_matrix = np.zeros((num_topics, num_topics))
        #hashtag_similarity_matrix[:] = np.nan
        #summary_and_hashtag_similarity_matrix = np.zeros((num_topics, num_topics))
        #summary_and_hashtag_similarity_matrix[:] = np.nan

        for row in range(num_topics):
            for col in range(row, num_topics):
                # Calculate cosine similarity between the two vectors
                summary_similarity = 1- cosine(summary_embeds[row], summary_embeds[col])
                summary_similarity_matrix[row, col] = summary_similarity
                summary_similarity_matrix[col, row] = summary_similarity
                #hashtag_similarity = 1- cosine(hashtag_embeds[row], hashtag_embeds[col])
                #hashtag_similarity_matrix[row, col] = hashtag_similarity
                #hashtag_similarity_matrix[col, row] = hashtag_similarity
                #summary_and_hashtag_similarity = 1- cosine(summary_and_hashtag_embeds[row], summary_and_hashtag_embeds[col])
                #summary_and_hashtag_similarity_matrix[row, col] = summary_and_hashtag_similarity
                #summary_and_hashtag_similarity_matrix[col, row] = summary_and_hashtag_similarity
        # Draw a heatmap with the summary_similarity_matrix
        #plt.figure()
        # Color scheme blues
        #plt.imshow(summary_similarity_matrix, cmap = 'Blues')
        #plt.imshow(hashtag_similarity_matrix, cmap = 'Reds')
        target_num_topics = 10
        topics_out_summaries = self._group_topics(title_similarity=summary_similarity_matrix, num_topics=target_num_topics, bonus_constant=0.2)
        #topics_out_hashtags = self._group_topics(title_similarity=hashtag_similarity_matrix, num_topics=target_num_topics, bonus_constant=0.2)
        #topics_out_summaries_and_hashtags = self._group_topics(title_similarity=summary_and_hashtag_similarity_matrix, num_topics=target_num_topics, bonus_constant=0.2)
        chunk_topics_summaries = topics_out_summaries['chunk_topics']
        topics_groupings_summaries = topics_out_summaries['topics']
        #chunk_topics_hashtags = topics_out_hashtags['chunk_topics']
        #topics_groupings_hashtags = topics_out_hashtags['topics']
        #chunk_topics_summaries_and_hashtags = topics_out_summaries_and_hashtags['chunk_topics']
        #topics_groupings_summaries_and_hashtags = topics_out_summaries_and_hashtags['topics']
        print('Suggested topic groupings by summary: ' + str(topics_groupings_summaries))
        #print('Suggested topic groupings by hashtag: ' + str(topics_groupings_hashtags))
        #print('Suggested topic groupings by summary & hashtag: ' + str(topics_groupings_summaries_and_hashtags))
        # Plot a heatmap of this array
        #plt.figure(figsize = (10, 4))
        #plt.imshow(np.array(chunk_topics_summaries).reshape(1, -1), cmap = 'tab20')
        # Draw vertical black lines for every 1 of the x-axis 
        #for i in range(1, len(chunk_topics_summaries)):
            #plt.axvline(x = i - 0.5, color = 'black', linewidth = 0.5)
            
        return topics_groupings_summaries
    
    def summarize_groups(self, stage_1_outputs, topics, summary_num_words=500):
        start = datetime.now()
        print(f"summarize_groups start time {start}")

        # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
        title_prompt_template = """Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible, 
        and do not overlap with each other. Add the most relevant emoji to the start of each title:
        {text}
        
        Return your answer in a numbered list, with a new line separating each title as in these examples: 
        1. ❄️ Title 1
        2. 🏆 Title 2
        3. 🚀 Title 3

        TITLES:
        """

        map_prompt_template = """Write a roughly 100-word summary of the following text:
            {text}

            CONCISE SUMMARY:"""

        combine_prompt_template = (
            "Write a "
            + str(summary_num_words)
            + """-word summary of the following, removing irrelevant information. Finish your answer:
        {text}
        """
            + str(summary_num_words)
            + """-WORD SUMMARY:"""
        )

        title_prompt = PromptTemplate(
            template=title_prompt_template, input_variables=["text"]
        )
        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )
        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        topics_data = []
        for c in topics:
            topic_data = {
                "summaries": [stage_1_outputs[chunk_id]["topic_summary"] for chunk_id in c],
                "titles": [stage_1_outputs[chunk_id]["topic_title"] for chunk_id in c],
            }
            topic_data["summaries_concat"] = " ".join(topic_data["summaries"])
            topic_data["titles_concat"] = ", ".join(topic_data["titles"])
            topics_data.append(topic_data)

        # Get a list of each community's summaries (concatenated)
        topics_summary_concat = [c["summaries_concat"] for c in topics_data]
        topics_titles_concat = [c["titles_concat"] for c in topics_data]
        print('Suggested topic groupings: ' + str(topics_titles_concat))

        # Concat into one long string to do the topic title creation
        topics_titles_concat_all = """"""
        for i, c in enumerate(topics_titles_concat):
            topics_titles_concat_all += f"""{i+1}. {c}
            """

        # print('topics_titles_concat_all', topics_titles_concat_all)

        title_llm = OpenAI(temperature=0, model_name=CONSOLIDATE_TITLES_MODEL)
        title_llm_chain = LLMChain(llm=title_llm, prompt=title_prompt)
        title_llm_chain_input = [{"text": topics_titles_concat_all}]
        title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)

        # Split by new line
        titles = title_llm_chain_results[0]["text"].split("\n")
        # Remove any empty titles
        titles = [t.strip() for t in titles if t.strip() != ""]

        map_llm = OpenAI(temperature=0, model_name=MAP_REDUCE_TOPICS_MODEL)
        reduce_llm = OpenAI(temperature=0, model_name=MAP_REDUCE_TOPICS_MODEL, max_tokens=-1)

        # Run the map-reduce chain
        docs = [Document(page_content=t) for t in topics_summary_concat]
        chain = load_summarize_chain(
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            return_intermediate_steps=True,
            llm=map_llm,
            reduce_llm=reduce_llm,
        )

        output = chain({"input_documents": docs}, return_only_outputs=True)
        summaries = output["intermediate_steps"]
        stage_2_outputs = [
            {"topic_title": t, "topic_summary": s.replace("\n","").strip()} for t, s in zip(titles, summaries)
        ]
        final_summary = output["output_text"]

        # Return: stage_1_outputs (title and summary), stage_2_outputs (title and summary), final_summary, chunk_allocations
        out = {"stage_2_outputs": stage_2_outputs, "final_summary": final_summary}
        end = datetime.now()
        duration = int((end - start).total_seconds())
        print(f"summarize_groups completed time {end}\nDuration: {duration} seconds")

        return out