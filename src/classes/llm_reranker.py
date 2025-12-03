#!/usr/bin/env python3
"""
LLM-based reranker using LangChain + Ollama (defaults to llama3.1:8b-instruct-q8_0-16k).
Author: Moniruzzaman Akash


Make sure you have the ollama server running locally:(for details follow README.md)
cd <project_root>
ollama pull llama3.1:8b-instruct-q8_0
ollama create llama3.1:8b-instruct-q8_0-16k -f src/getting_started/Modelfile
ollama serve


Example Usage:
reranker = LLMReranker()  # uses OLLAMA_MODEL or defaults to "llama3.1:8b-instruct-q8_0-16k"
example_queries = ["example query 1", "example query 2"]
reranker.rerank_runfile(
    initial_run_path="../../run_files/tfidf_lnc_nnn_ThesaurusExp.run", # Initial run file path before reranking
    corpus_path     ="../../data/corpus_jsonl/corpus.jsonl",           # Corpus file path in JSONL format
    output_run_path ="../../run_files/tfidf_lnc_nnn_ThesaurusExp_LLM-Rerank.run", # Output run file path after reranking
    queries         = example_queries,                               # List of query strings
    log_path        ="../../logs/llm_reranker_logs.jsonl",           # Optional log file path for prompts/responses
    top_k           =20,                                             # Number of top documents to rerank per query
    run_tag         ="llm_reranker",                                 # Tag to use in the output run file
)
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from datasets import load_from_disk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tqdm import tqdm


class LLMReranker:
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        num_predict: int = 512,
    ):
        # Resolve model: environment override -> provided -> default quantized tag
        resolved_model = (
            os.environ.get("OLLAMA_MODEL")
            or model_name
            or "llama3.1:8b-instruct-q8_0-16k"
        )

        print(f"Using Ollama model: {resolved_model}")
        # LangChain -> Ollama chat model
        self.llm = ChatOllama(
            model=resolved_model,
            temperature=temperature,
            num_predict=num_predict,
        )

        # Listwise ranking prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an information retrieval ranking model. "
                    "Given a user query and a set of candidate documents, "
                    "you assign each document a relevance score between 0 and 10.\n"
                    "- 10 = highly relevant\n"
                    "- 5  = somewhat relevant\n"
                    "- 1  = marginally relevant\n"
                    "- 0  = not relevant\n"
                    "Return ONLY valid JSON. No explanations and comments.",
                ),
                (
                    "user",
                    (
                        "Query:\n"
                        "{query}\n\n"
                        "Candidate documents (each has an id and text):\n"
                        "{documents}\n\n"
                        "Respond ONLY with a JSON array of objects of the form:\n"
                        '[{{"docid": "D1", "score": 3}}, {{"docid": "D2", "score": 1}}, ...]\n'
                        "You must include an entry for every document id that appears above."
                    ),
                ),
            ]
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def _messages_to_text(messages: List[Any]) -> str:
        lines = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", "message")
            lines.append(f"{role.upper()}: {m.content}")
        return "\n".join(lines)

    @staticmethod
    def _format_docs(docs: List[Dict[str, Any]]) -> str:
        """
        Turn a list of {id, text} into a compact, LLM-friendly string.
        """
        lines = []
        for i, d in enumerate(docs, start=1):
            doc_id = d["id"]
            text = d["text"].replace("\n", " ")
            lines.append(f"{i}. [docid={doc_id}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _extract_json(response: str) -> Any:
        """
        Extract JSON array from the LLM response, handling:
        - ```json ... ``` fences
        - extra text before/after
        - stray // line comments and trailing commas
        - malformed arrays by falling back to regex extraction of docid/score pairs
        """
        # Strip code fences if present
        fence_match = re.search(r"```json(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
        else:
            candidate = response.strip()

        # Try to locate first '[' and last ']' to isolate array
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]

        def _clean(text: str) -> str:
            # remove // comments
            text = re.sub(r"//.*", "", text)
            # remove trailing commas before ] or }
            text = re.sub(r",\s*([\]}])", r"\1", text)
            return text

        try:
            return json.loads(candidate)
        except Exception:
            cleaned = _clean(candidate)
            try:
                return json.loads(cleaned)
            except Exception:
                # Fallback: extract objects with docid/score via regex
                objects = []
                seen = set()
                for m in re.finditer(r'\{[^}]*?\}', cleaned):
                    try:
                        obj = json.loads(_clean(m.group(0)))
                        docid = str(obj.get("docid"))
                        if docid in seen:
                            continue
                        seen.add(docid)
                        objects.append({"docid": docid, "score": obj.get("score", 0)})
                    except Exception:
                        continue
                if objects:
                    return objects
                raise

    def rerank(
        self, query: str, docs: List[Dict[str, Any]], return_raw: bool = False
    ) -> Any:
        """
        Run LLM reranking.

        Args:
            query: user query string
            docs: list of dicts like {"id": "D1", "text": "..."}
            return_raw: if True, also return rendered prompt and raw LLM response
        Returns:
            List of (doc_id, score, text) sorted by score (desc)
            If return_raw is True, returns (results, prompt_text, raw_response)
        """
        if not docs:
            return []

        try:
            messages = self.prompt.format_messages(
                query=query,
                documents=self._format_docs(docs),
            )
            raw_message = self.llm.invoke(messages)
            raw_response = raw_message.content if hasattr(raw_message, "content") else str(
                raw_message
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Failed to reach the local Ollama server. Make sure ollama is running "
                "and the model is available (e.g., `ollama serve` then `ollama pull "
                f"{self.llm.model}`), or set OLLAMA_HOST if it runs elsewhere."
            ) from exc

        try:
            scored_list = self._extract_json(raw_response)
        except Exception as e:
            # Gracefully skip this rerank attempt on parse failures
            print(f"Warning: failed to parse LLM JSON output ({e}); skipping this query.")
            if return_raw:
                return [], self._messages_to_text(messages), raw_response
            return []

        # Build a map from id -> text for convenience
        id_to_text = {d["id"]: d["text"] for d in docs}

        results = []
        for item in scored_list:
            docid = str(item.get("docid"))
            score = float(item.get("score", 0.0))
            if docid in id_to_text:
                results.append((docid, score, id_to_text[docid]))

        # Sort by score (desc), tie-break by docid for determinism
        results.sort(key=lambda x: (-x[1], x[0]))
        if return_raw:
            return results, self._messages_to_text(messages), raw_response
        return results

    def rerank_runfile(
        self,
        initial_run_path: str,
        corpus_path: str,
        output_run_path: str,
        queries: Optional[List[str]] = None,
        log_path: Optional[str] = None,
        top_k: int = 20,
        run_tag: str = "llm_reranker",
    ) -> None:
        """
        Rerank top-k docs from an existing run file and write a new run file.
        Also streams prompts/responses to a jsonl log if log_path is provided.
        If queries is provided, it should be a list where index aligns to the qid
        in the run file (0-based or whatever the run uses).
        """
        # Parse run file -> per-query top_k docids
        qid_to_docids: Dict[int, List[str]] = {}
        needed_docids: set[str] = set()
        with open(initial_run_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    qid = int(parts[0])
                except ValueError:
                    continue
                docid = parts[2]
                slot = qid_to_docids.setdefault(qid, [])
                if len(slot) < top_k:
                    slot.append(docid)
                    needed_docids.add(docid)

        # Load corpus texts only for needed docids
        doc_texts: Dict[str, str] = {}
        with open(corpus_path, "r", encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                docid = str(obj.get("id"))
                if docid in needed_docids:
                    doc_texts[docid] = obj.get("contents", "")
                if len(doc_texts) == len(needed_docids):
                    break

        # Prepare logging
        log_f = open(log_path, "w", encoding="utf8") if log_path else None

        with open(output_run_path, "w", encoding="utf8") as out_f:
            for qid in tqdm(sorted(qid_to_docids.keys()), desc="LLM reranking"):
                docids = qid_to_docids[qid]
                if queries:
                    query_text = queries[qid] if qid < len(queries) else ""
                else:
                    query_text = ""

                docs = []
                for d in docids:
                    text = doc_texts.get(str(d), "")
                    if text:
                        docs.append({"id": str(d), "text": text})

                if not docs:
                    continue

                results, prompt_text, raw_response = self.rerank(
                    query_text, docs, return_raw=True
                )
                
                if len(results) == 0:
                    continue  # Skip if no valid results
                for rank, (docid, score, _) in enumerate(results, start=1):
                    out_f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_tag}\n")

                if log_f:
                    log_f.write(
                        json.dumps(
                            {
                                "qid": qid,
                                "query": query_text,
                                "prompt": prompt_text,
                                "response": raw_response,
                                "results": results,
                            }
                        )
                        + "\n"
                    )

        if log_f:
            log_f.close()


def main():
    # Example usage
    
    print(""" Example LLM reranker usage: \n
    reranker = LLMReranker()  # uses OLLAMA_MODEL or defaults to "llama3.1:8b-instruct-q8_0-16k"
    example_queries = ["example query 1", "example query 2"]
    reranker.rerank_runfile(
    initial_run_path="../../run_files/tfidf_lnc_nnn_ThesaurusExp.run", # Initial run file path before reranking
    corpus_path     ="../../data/corpus_jsonl/corpus.jsonl",           # Corpus file path in JSONL format
    output_run_path ="../../run_files/tfidf_lnc_nnn_ThesaurusExp_LLM-Rerank.run", # Output run file path after reranking
    queries         = example_queries,                               # List of query strings aligned with qids
    log_path        ="../../logs/llm_reranker_logs.jsonl",           # Optional log file path for prompts/responses
    top_k           =20,                                             # Number of top documents to rerank per query
    run_tag         ="llm_reranker",                                 # Tag to use in the output run file
          
    """)
    


if __name__ == "__main__":
    main()
