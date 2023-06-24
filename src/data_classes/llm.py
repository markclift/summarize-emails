import dataclasses
from config import MODEL_CONFIGS

@dataclasses.dataclass
class LLM:
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    config: dict = dataclasses.field(init=False)

    def __post_init__(self):
        self.config = MODEL_CONFIGS.get(self.model_name)

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