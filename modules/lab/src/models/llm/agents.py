import os
import sys
import time
from abc import ABC, abstractmethod

from openai import OpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams


EINFRA_MODELS = [
    "gpt-oss-120b",
    "phi4:14b-q8_0",
    "mistral-small3.2:24b-instruct-2506-q8_0",
    "llama-4-scout-17b-16e-instruct",
]

LOCAL_MODELS = [
    # "google/gemma-3n-E2B", does not work on gpu server
    "google/gemma-3-1b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "microsoft/Phi-4-mini-reasoning",
    "mistralai/Ministral-3-3B-Reasoning-2512",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
]


class ModelAgent(ABC):
    """Abstract base class for model agents."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
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
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs,
    ) -> str:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        outputs = self.llm.generate([full_prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs,
    ) -> list[str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        full_prompts = []
        for prompt in prompts:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            full_prompts.append(full_prompt)

        outputs = self.llm.generate(full_prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


class APIAgent(ModelAgent):
    """Agent for online API models using OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        use_batch_api: bool = False,
    ):
        self.model_name = model_name
        self.use_batch_api = use_batch_api
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs,
    ) -> list[str]:
        if self.use_batch_api:
            return self._generate_batch_api(
                prompts, system_prompt, temperature, max_tokens
            )
        else:
            return self._generate_batch_sequential(
                prompts, system_prompt, temperature, max_tokens
            )

    def _generate_batch_sequential(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> list[str]:
        """Process prompts sequentially using individual API calls."""
        responses = []
        for prompt in tqdm(prompts, desc="Processing prompts", file=sys.stdout):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            responses.append(response.choices[0].message.content.strip())
        return responses

    def _generate_batch_api(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> list[str]:
        """Process prompts using OpenAI Batch API (asynchronous, discounted pricing)."""
        requests = []
        for i, prompt in enumerate(prompts):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            requests.append(
                {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
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
                results_by_id[custom_id] = result.result.message.content.strip()
            else:
                results_by_id[custom_id] = ""

        return [results_by_id[i] for i in range(len(prompts))]
