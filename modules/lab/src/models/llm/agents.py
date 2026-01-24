from dataclasses import dataclass
import os
import sys
import time
from abc import ABC, abstractmethod

from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from models.llm.prompts import schema_for_suffix_key

MAX_TOKENS = 1024


def _create_retry_decorator():
    """Create a retry decorator for API calls with exponential backoff.

    Retries on rate limits and temporary API errors, with exponential backoff.
    Max 5 attempts: wait 2s, 4s, 8s, 16s between retries.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=32),
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        reraise=True,
    )


def _choose_schema(
    structured_schema: dict | None, default_key: str | None = None
) -> dict | None:
    if structured_schema:
        return structured_schema
    if default_key:
        return schema_for_suffix_key(default_key)
    return None


@dataclass
class ModelConfig:
    id: str
    name: str
    params: SamplingParams
    system_prefix: str | None = None
    system_suffix: str | None = None
    prompt_prefix: str | None = None
    prompt_suffix: str | None = None
    enable_thinking: bool = False
    llm_args: dict | None = None


EINFRA_MODELS = [
    # "phi4:14b-q8_0", bad
    ModelConfig(
        id="gpt-oss-120b",
        name="GPT-oss-120b",
        params=SamplingParams(temperature=1.0),
        system_prefix="<|start|>system<|message|>",
        system_suffix="<|end|>",
        prompt_prefix="<|start|>user<|message|>",
        prompt_suffix="<|end|>\n<|start|>assistant<|message|>",
    ),
    # ModelConfig(
    #     id="mistral-small3.2:24b-instruct-2506-q8_0",
    #     name="Mistral-Small3.2-24b",
    #     params=SamplingParams(temperature=0.15, top_p=1.0),
    #     system_prefix="<s>[SYSTEM_PROMPT]",
    #     system_suffix="[/SYSTEM_PROMPT]",
    #     prompt_prefix="[INST]",
    #     prompt_suffix="[/INST]",
    # ),
    # ModelConfig(
    #     id="llama-4-scout-17b-16e-instruct",
    #     name="Llama4-Scout-17b",
    #     enable_thinking=False,
    #     params=SamplingParams(temperature=1.0),
    #     system_prefix="<|header_start|>system<|header_end|>\n\n",
    #     system_suffix="<|eot|>",
    #     prompt_prefix="<|header_start|>user<|header_end|>\n\n",
    #     prompt_suffix="<|eot|>\n<|header_start|>assistant<|header_end|>\n\n",
    # ),
]

LOCAL_MODELS = [
    # "google/gemma-3n-E2B", does not work on gpu server
    # "google/gemma-3-1b-it", bad
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", bad
    # "microsoft/Phi-4-mini-reasoning", bad
    # "Qwen/Qwen3-0.6B", bad
    # "Qwen/Qwen3-1.7B", bad
    # "mistralai/Ministral-3-3B-Reasoning-2512",
    # https://unsloth.ai/docs/models/ministral-3
    ModelConfig(
        id="unsloth/Ministral-3-14B-Instruct-2512-FP8",
        name="Ministral3-14b",
        params=SamplingParams(temperature=0.15),
        system_prefix="<s>[SYSTEM_PROMPT]",
        system_suffix="[/SYSTEM_PROMPT]",
        prompt_prefix="[INST]",
        prompt_suffix="[/INST]",
    ),
    # https://unsloth.ai/docs/models/gemma-3-how-to-run-and-fine-tune
    ModelConfig(
        id="unsloth/gemma-3-12b-it-bnb-4bit",
        name="Gemma3-12b",
        params=SamplingParams(
            temperature=1.0, top_k=64, min_p=0.01, top_p=0.95, repetition_penalty=1.0
        ),
        system_prefix="<start_of_turn>system\n",
        system_suffix="<end_of_turn>\n",
        prompt_prefix="<start_of_turn>user\n",
        prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
    ),
    # https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune
    ModelConfig(
        id="unsloth/Qwen3-14B-FP8",
        name="Qwen3-14b",
        enable_thinking=False,
        params=SamplingParams(temperature=0.7, top_k=20, min_p=0.01, top_p=0.8),
        system_prefix="<|im_start|>system\n",
        system_suffix="<|im_end|>\n",
        prompt_prefix="<|im_start|>user\n",
        prompt_suffix="<|im_end|>\n<|im_start|>assistant\n",
    ),
    # https://unsloth.ai/docs/models/nemotron-3
    # ModelConfig(
    #     id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    #     name="Nemotron3Nano-30b",
    #     enable_thinking=False,
    #     params=SamplingParams(temperature=1.0, top_p=1.0),
    #     system_prefix="<|im_start|>system\n",
    #     system_suffix="<|im_end|>\n",
    #     prompt_prefix="<|im_start|>user\n",
    #     prompt_suffix="<|im_end|>\n<|im_start|>assistant\n",
    # ),
    # "meta-llama/Llama-3.2-3B-Instruct",
    # https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct
    # ModelConfig(
    #     id="unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
    #     name="llama3.2-11b",
    #     enable_thinking=False,
    #     params=SamplingParams(temperature=1.0),
    #     system_prefix="<|header_start|>system<|header_end|>\n\n",
    #     system_suffix="<|eot|>",
    #     prompt_prefix="<|header_start|>user<|header_end|>\n\n",
    #     prompt_suffix="<|eot|>\n<|header_start|>assistant<|header_end|>\n\n",
    # ), does not work in vLLM
    ModelConfig(
        id="jiangchengchengNLP/Mistral-Small-3.2-24B-Instruct-W8A8",
        name="Mistral-Small3.2-24b",
        params=SamplingParams(temperature=0.15, top_p=1.0),
        system_prefix="<s>[SYSTEM_PROMPT]",
        system_suffix="[/SYSTEM_PROMPT]",
        prompt_prefix="[INST]",
        prompt_suffix="[/INST]",
    ),
    ModelConfig(
        id="RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
        name="Llama4-Scout-17b",
        enable_thinking=False,
        params=SamplingParams(temperature=0.7, top_p=0.8),
        system_prefix="<|header_start|>system<|header_end|>\n\n",
        system_suffix="<|eot|>",
        prompt_prefix="<|header_start|>user<|header_end|>\n\n",
        prompt_suffix="<|eot|>\n<|header_start|>assistant<|header_end|>\n\n",
    ),
]

AVAILABLE_MODELS = EINFRA_MODELS + LOCAL_MODELS
MODEL_BY_ID = {model.id: model for model in AVAILABLE_MODELS}
MODEL_BY_NAME = {model.name: model for model in AVAILABLE_MODELS}


class ModelAgent(ABC):
    """Abstract base class for model agents."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def generate_batch(
        self, prompts: list[str], system_prompt: str | None = None, **kwargs
    ) -> list[str]:
        """Generate text from multiple prompts."""
        pass


class VLLMAgent(ModelAgent):
    """Agent for local models using vLLM."""

    def __init__(
        self,
        model_config: "ModelConfig",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        self.model_config = model_config
        self.model_name = model_config.name
        self.llm = LLM(
            model=model_config.id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            **(model_config.llm_args or {}),
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_structured_outputs: bool = True,
        structured_schema: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> str:
        chosen_schema = _choose_schema(structured_schema, "cot")

        base_params = self.model_config.params
        sampling_kwargs = {
            "temperature": base_params.temperature,
            "max_tokens": MAX_TOKENS,
            "top_p": base_params.top_p,
            "top_k": base_params.top_k,
            "min_p": base_params.min_p,
            "repetition_penalty": base_params.repetition_penalty,
        }

        if seed is not None:
            sampling_kwargs["seed"] = seed

        if use_structured_outputs and chosen_schema:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=chosen_schema
            )

        sampling_params = SamplingParams(**sampling_kwargs)

        full_prompt = prompt
        if self.model_config.system_prefix and system_prompt:
            full_prompt = f"{self.model_config.system_prefix}{system_prompt}{self.model_config.system_suffix}{self.model_config.prompt_prefix}{prompt}{self.model_config.prompt_suffix}"
        elif self.model_config.prompt_prefix:
            full_prompt = f"{self.model_config.prompt_prefix}{prompt}{self.model_config.prompt_suffix}"

        outputs = self.llm.generate([full_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        use_structured_outputs: bool = True,
        structured_schema: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[str]:
        chosen_schema = _choose_schema(structured_schema, "cot")

        base_params = self.model_config.params
        sampling_kwargs = {
            "temperature": base_params.temperature,
            "max_tokens": MAX_TOKENS,
            "top_p": base_params.top_p,
            "top_k": base_params.top_k,
            "min_p": base_params.min_p,
            "repetition_penalty": base_params.repetition_penalty,
        }

        if seed is not None:
            sampling_kwargs["seed"] = seed

        if use_structured_outputs and chosen_schema:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
                json=chosen_schema
            )

        sampling_params = SamplingParams(**sampling_kwargs)

        full_prompts = []
        for prompt in prompts:
            full_prompt = prompt
            if self.model_config.system_prefix and system_prompt:
                full_prompt = f"{self.model_config.system_prefix}{system_prompt}{self.model_config.system_suffix}{self.model_config.prompt_prefix}{prompt}{self.model_config.prompt_suffix}"
            elif self.model_config.prompt_prefix:
                full_prompt = f"{self.model_config.prompt_prefix}{prompt}{self.model_config.prompt_suffix}"
            full_prompts.append(full_prompt)

        outputs = self.llm.generate(full_prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


class APIAgent(ModelAgent):
    """Agent for online API models using OpenAI-compatible API."""

    def __init__(
        self,
        model_config: "ModelConfig",
        api_key: str | None = None,
        base_url: str | None = None,
        use_batch_api: bool = False,
    ):
        self.model_config = model_config
        self.model_id = model_config.id
        self.model_name = model_config.name
        self.use_batch_api = use_batch_api
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    @_create_retry_decorator()
    def _call_chat_completions(
        self, messages: list, response_format=None, seed: int | None = None
    ) -> str:
        """Call chat completions API with automatic retry on transient errors."""
        kwargs = {
            "model": self.model_id,
            "temperature": self.model_config.params.temperature,
            "max_tokens": MAX_TOKENS,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        use_structured_outputs: bool = True,
        structured_schema: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_format = (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "LLMRating",
                    "schema": _choose_schema(structured_schema, "cot"),
                },
            }
            if use_structured_outputs and _choose_schema(structured_schema, "cot")
            else None
        )

        return self._call_chat_completions(messages, response_format, seed=seed)

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        use_structured_outputs: bool = True,
        structured_schema: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> list[str]:
        if self.use_batch_api:
            return self._generate_batch_api(
                prompts,
                system_prompt,
                use_structured_outputs=use_structured_outputs,
                structured_schema=structured_schema,
                seed=seed,
            )
        else:
            return self._generate_batch_sequential(
                prompts,
                system_prompt,
                use_structured_outputs=use_structured_outputs,
                structured_schema=structured_schema,
                seed=seed,
            )

    def _generate_batch_sequential(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        use_structured_outputs: bool = True,
        structured_schema: dict | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Process prompts sequentially using individual API calls."""
        responses = []
        for prompt in tqdm(prompts, desc="Processing prompts", file=sys.stdout):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response_format = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMRating",
                        "schema": _choose_schema(structured_schema, "cot"),
                    },
                }
                if use_structured_outputs and _choose_schema(structured_schema, "cot")
                else None
            )

            content = self._call_chat_completions(messages, response_format, seed=seed)
            responses.append(content)
        return responses

    def _generate_batch_api(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        use_structured_outputs: bool = True,
        structured_schema: dict | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Process prompts using OpenAI Batch API (asynchronous, discounted pricing)."""
        requests = []
        for i, prompt in enumerate(prompts):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body = {
                "model": self.model_id,
                "messages": messages,
                "temperature": self.model_config.params.temperature,
                "max_tokens": MAX_TOKENS,
            }

            response_format = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LLMRating",
                        "schema": _choose_schema(structured_schema, "cot"),
                    },
                }
                if use_structured_outputs and _choose_schema(structured_schema, "cot")
                else None
            )
            if response_format is not None:
                body["response_format"] = response_format
            if seed is not None:
                body["seed"] = seed

            requests.append(
                {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )

        batch = self.client.beta.batches.create(requests=requests)
        batch_id = batch.id
        print(f"          Batch {batch_id} created. Polling for completion...")

        while batch.status not in ["completed", "failed", "expired"]:
            print(f"          Batch status: {batch.status}... ", end="", flush=True)
            time.sleep(2)
            batch = self.client.beta.batches.retrieve(batch_id)
            print("✓")

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch_id} failed with status: {batch.status}")
        print("          Batch completed successfully.")

        results_by_id = {}
        for result in self.client.beta.batches.results(batch_id):
            custom_id = int(result.custom_id)
            if result.result.message:
                content = result.result.message.content
                results_by_id[custom_id] = content.strip() if content else ""
            else:
                results_by_id[custom_id] = ""

        return [results_by_id[i] for i in range(len(prompts))]
