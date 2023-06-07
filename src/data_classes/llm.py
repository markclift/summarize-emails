import dataclasses

GPT4_NAME = "gpt-4"
GPT35_NAME = "gpt-3.5-turbo"
GPT3_NAME = "text-davinci-003"

MODEL_CONFIGS = {
    GPT4_NAME: {"max_tokens_input": 5800, "max_tokens_response": 2000, "cost_per_1k_tokens_prompt": 0.03, "cost_per_1k_tokens_completion": 0.06},
    GPT35_NAME: {"max_tokens_input": 2800, "max_tokens_response": 1000, "cost_per_1k_tokens_prompt": 0.002, "cost_per_1k_tokens_completion": 0.002},
    GPT3_NAME: {"max_tokens_input": 2800, "max_tokens_response": 1000, "cost_per_1k_tokens_prompt": 0.02, "cost_per_1k_tokens_completion": 0.02},
    "default": {"max_tokens_input": 0, "max_tokens_response": 0, "cost_per_1k_tokens_prompt": 0, "cost_per_1k_tokens_completion": 0}
}


@dataclasses.dataclass
class LLM:
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    config: dict = dataclasses.field(init=False)

    def __post_init__(self):
        self.config = MODEL_CONFIGS.get(self.model_name, MODEL_CONFIGS["default"])

    def get_cost(self, prompt_tokens=None, completion_tokens=None):
        cost_per_1k_tokens_prompt = self.config["cost_per_1k_tokens_prompt"]
        cost_per_1k_tokens_completion = self.config["cost_per_1k_tokens_completion"]
        
        prompt_tokens = prompt_tokens if prompt_tokens is not None else self.prompt_tokens
        completion_tokens = completion_tokens if completion_tokens is not None else self.completion_tokens

        cost = (prompt_tokens / 1000 * cost_per_1k_tokens_prompt) + (
            completion_tokens / 1000 * cost_per_1k_tokens_completion
        )
        cost = round(cost, 2)  # Round the cost to 2 decimal places
        return cost