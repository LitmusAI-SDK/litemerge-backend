from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int  # provider-reported cache hit tokens (0 if unsupported)
    cache_write_tokens: int  # provider-reported cache write tokens (0 if unsupported)
