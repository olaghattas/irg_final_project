import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai._exceptions import (
    APIConnectionError, APITimeoutError, APIStatusError, RateLimitError, OpenAIError
)
from aiolimiter import AsyncLimiter
from langchain.prompts import PromptTemplate

MIN_BACKOFF = 2
MAX_BACKOFF = 30
MAX_ATTEMPTS = 6


class QueryExpansion:
    def __init__(self, path_to_api_key=None, openrouter_key_name="OPENROUTER_API_KEY"):

        if path_to_api_key is None:
            path_to_api_key = "/home/olagh48652/irg_course_assig/irg-programming/.env/irgprog.env"

        load_dotenv(dotenv_path=path_to_api_key)
        openrouter_api_key = os.environ[openrouter_key_name]

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        tmpl = """
        Expand the following query by adding relevant terms, entities, and concepts.
        Return only the expanded query as a single clean text line.
        Do not include formatting, markdown, bullets, or explanations.

        Original query:
        {query}
        """

        self.prompt_template = PromptTemplate.from_template(tmpl)

        # Allow max 20 requests/minute (OpenRouter limit for free models)
        self.rate_limiter = AsyncLimiter(max_rate=20, time_period=60)

        # Avoid bursts: only 1 request running at a time (safe for free tier)
        self.concurrent = asyncio.Semaphore(1)

    async def _ask_once(self, prompt: str) -> str:
        """Internal: Make a single LLM request with retries + rate limiting."""
        backoff = MIN_BACKOFF

        for attempt in range(1, MAX_ATTEMPTS + 1):
            async with self.concurrent:         # ensure NO parallel API calls
                async with self.rate_limiter:   # ensure <= 20 calls/min
                    try:
                        resp = await self.client.chat.completions.create(
                            model="gpt-oss-20b:free",
                            messages=[
                                {"role": "system", "content": "Produce an expanded version of the user's query."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.3,
                        )
                        return resp.choices[0].message.content

                    except (APIConnectionError, APITimeoutError, APIStatusError, RateLimitError) as e:
                        print(f"Transient error {attempt}: {e}. Retry in {backoff}s")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF)

                    except OpenAIError as e:
                        print(f"Permanent error: {e}")
                        raise

        raise RuntimeError("Max retries exceeded.")

    async def expand_queries(self, queries, save_path=None):
        if not isinstance(queries, list):
            raise TypeError("queries must be a list of strings.")
        if any(not isinstance(q, str) for q in queries):
            raise TypeError("each element in queries must be a string.")

        prompts = [self.prompt_template.format(query=q) for q in queries]
        print(f"Sending {len(prompts)} queries to LLM...")

        results = []

        # --- Batch size of 10 (but sequential requests inside each batch) ---
        for i in range(0, len(prompts), 20):
            batch_prompts = prompts[i:i + 20]
            batch_queries = queries[i:i + 20]

            for q_raw, prompt in zip(batch_queries, batch_prompts):
                expanded = await self._ask_once(prompt)
                results.append({
                    "original_query": q_raw.strip(),
                    "expanded_query": expanded.strip(),
                })

        # Optional save
        if save_path is not None:
            if save_path.endswith(".jsonl"):
                with open(save_path, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
            else:
                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2)

            print(f"Saved expansions to {save_path}")

        return results


async def main():
    qe = QueryExpansion()
    
    queries = [
        "what are transformer models",
        "neural ranking with bm25",
        "reinforcement learning for retrieval"
    ]
    
    exp = await qe.expand_queries(queries, save_path="expansions_test.jsonl")
    print(exp)

if __name__ == "__main__":
    asyncio.run(main())





