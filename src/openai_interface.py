import os
import openai
from dotenv import load_dotenv
from typing import List

class OpenAI_Interface():
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.MODEL="gpt-4"
        #self.MODEL="gpt-3.5-turbo"
        self.prompt_tokens=0
        self.completion_tokens=0

    def get_tokens_total(self):
        return self.prompt_tokens, self.completion_tokens

    def get_token_cost(self):
        if self.MODEL == "gpt-4":
            cost_per_1k_tokens_prompt = 0.03
            cost_per_1k_tokens_completion = 0.06
        elif self.MODEL == "gpt-3.5-turbo":
            cost_per_1k_tokens_prompt = 0.002
            cost_per_1k_tokens_completion = 0.002
        
        cost = (self.prompt_tokens/1000*cost_per_1k_tokens_prompt) + (self.completion_tokens/1000*cost_per_1k_tokens_completion)
        cost = round(cost, 2)  # Round the cost to 2 decimal places
        return cost

    def generate_summary(self, text_to_summarize: str, topics_of_interest: List[str]):
        system_prompt = "You understand technical topics and are able to explain them both succicently and in detail."
        
        user_prompt = f"Please summarize the following series of articles of events from the past 7 days. I am particularly interested in the following topics: {topics_of_interest} and so where some information relates to these topics, please explain it in detail. If it is not directly related, you can summarise it succicently:\n\n" + text_to_summarize
        
        gpt_message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            response = openai.ChatCompletion.create(
                model=self.MODEL,
                messages=gpt_message,
                max_tokens=800,
                temperature=0,
            )

            # extract the response content
            print(response)
            response_content = response['choices'][0]['message']['content'].strip()
            self.prompt_tokens+=response['usage']['prompt_tokens']
            self.completion_tokens+=response['usage']['completion_tokens']

            return response_content

        except Exception as error:
            raise error