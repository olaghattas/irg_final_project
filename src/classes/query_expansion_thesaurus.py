import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

from tqdm import tqdm


try:
    import nltk
    from nltk.corpus import wordnet as wn

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    wn = None


class QueryExpansionThesaurus:
    """
    Thesaurus-based query expansion using WordNet.

    Example:
        qe = QueryExpansionThesaurus()
        expansions = qe.expand_queries(["graph neural networks for retrieval"])
    """

    def __init__(
        self,
        max_synonyms_per_word: int = 3,
        min_word_length: int = 1,
        allowed_pos: Optional[Iterable[str]] = ("n", "v", "a", "r"),
        stopwords_path: Optional[str] = None,
        download_if_missing: bool = True,
    ) -> None:
        """
        Args:
            max_synonyms_per_word: Limit to avoid overly long expansions.
            min_word_length: Skip very short tokens (e.g., stop-words).
            allowed_pos: Filter WordNet parts-of-speech (n, v, a, r). Set to None to allow all.
            stopwords_path: Path to a newline-delimited stopwords file. Defaults to stopwords.txt next to this module.
            download_if_missing: If True, attempt nltk.download('wordnet') when data is missing.
        """
        print("Initializing...")
        self.max_synonyms_per_word = max_synonyms_per_word
        self.min_word_length = min_word_length
        self.allowed_pos = set(allowed_pos) if allowed_pos else None
        self.stopwords = self._load_stopwords(stopwords_path)
        self.wordnet_ready = self._ensure_wordnet(download_if_missing)

    def _ensure_wordnet(self, download_if_missing: bool) -> None:
        if not _NLTK_AVAILABLE:
            print(
                "Warning: nltk is not installed; thesaurus expansion will return original queries only. "
                "Install with `pip install nltk` (or conda) and ensure network access to download 'wordnet'."
            )
            return False

        try:
            wn.synsets("test")
            return True
        except LookupError:
            if download_if_missing:
                print("Downloading NLTK WordNet corpus...")
                nltk.download("wordnet", quiet=True)
                try:
                    wn.synsets("test")  # re-check
                    return True
                except LookupError:
                    print(
                        "Warning: WordNet download failed; thesaurus expansion will return original queries only."
                    )
                    return False
            else:
                raise RuntimeError(
                    "NLTK WordNet corpus not found. Run nltk.download('wordnet') once or set "
                    "download_if_missing=True."
                )

    def _load_stopwords(self, path: Optional[str]) -> Set[str]:
        if path is None:
            path = Path(__file__).with_name("stopwords.txt")
        else:
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Stopwords file not found at {path}. Provide stopwords_path or create this file."
            )
        with open(path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return set(words)

    def _synonyms_for_word(self, word: str) -> List[str]:
        if not self.wordnet_ready:
            return []
        syns = []
        for synset in wn.synsets(word):
            if self.allowed_pos and synset.pos() not in self.allowed_pos:
                continue
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower():
                    syns.append(candidate)
            if len(syns) >= self.max_synonyms_per_word:
                break
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in syns:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)
            if len(unique) >= self.max_synonyms_per_word:
                break
        return unique

    def expand_query(self, query: str) -> str:
        """
        Expand a single query using WordNet synonyms.
        Returns the original query plus appended synonyms (space-separated).
        """
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']*", query)
        expansions = []

        for tok in tokens:
            if len(tok) < self.min_word_length:
                continue
            if tok.lower() in self.stopwords:
                continue
            expansions.extend(self._synonyms_for_word(tok))

        # Combine original query with synonyms; preserve readability.
        if expansions:
            return query.strip() + " " + " ".join(expansions)
        return query.strip()

    def expand_queries(self, queries: List[str], save_path: Optional[str] = None):
        '''
        Expects a list of query strings, expands each using the thesaurus,
        and optionally saves the results to a file.
        '''
        if not isinstance(queries, list) or any(not isinstance(q, str) for q in queries):
            raise TypeError("queries must be a list of strings.")

        iterator = tqdm(queries, desc="Expanding Queries", unit="q")

        results = []
        for q in iterator:
            expanded = self.expand_query(q)
            results.append({"original_query": q.strip(), "expanded_query": expanded})

        if save_path is not None:
            if save_path.endswith(".jsonl"):
                with open(save_path, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
            else:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    # queries = [
    #     "graph neural networks for retrieval",
    #     "question answering systems",
    #     "document ranking",
    # ]
    # current_file_path = Path(__file__).parent
    # qe = QueryExpansionThesaurus(max_synonyms_per_word=3, stopwords_path="dataset/stopwords.txt")
    # results = qe.expand_queries(queries, save_path="dataset/expansions_thesaurus.jsonl")
    # print(f"Saved {len(results)} expanded queries to expansions_thesaurus.jsonl")
    pass
