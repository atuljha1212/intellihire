"""spaCy loader plus the tokenisation helpers used by the rest of the NLP layer.

The trained `en_core_web_sm` pipeline (lemmatiser, POS tagger, parser, NER) is
used when installed; otherwise a blank English tokenizer keeps the phrase
matcher working, and a regex fallback is used if spaCy itself is missing.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from functools import lru_cache

from intellihire.config import get_settings

log = logging.getLogger(__name__)

try:
    import spacy

    HAS_SPACY = True
except ImportError:  # pragma: no cover - depends on environment
    spacy = None
    HAS_SPACY = False

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "to", "was", "were", "will", "with", "we", "you", "our", "your", "i", "my",
    "me", "this", "these", "those", "they", "their", "he", "she", "his", "her", "who", "which", "what", "can",
    "able", "using", "use", "used", "etc", "also", "into", "over", "per", "via", "such", "other", "more", "than",
    "including", "across", "within", "well", "strong", "good", "work", "working", "experience", "years", "year",
}


@lru_cache(maxsize=1)
def get_nlp():
    """Return (nlp, status) where status describes which pipeline is active."""
    if not HAS_SPACY:
        return None, "unavailable (regex fallback)"
    name = get_settings().spacy_model
    try:
        nlp = spacy.load(name, disable=["textcat"])
        return nlp, f"{name}"
    except OSError:
        log.warning("spaCy model %s not installed; using blank English tokenizer", name)
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp, "blank 'en' tokenizer (run: python -m spacy download en_core_web_sm)"


def nlp_status() -> str:
    return get_nlp()[1]


def has_trained_pipeline() -> bool:
    nlp, _ = get_nlp()
    return nlp is not None and "parser" in nlp.pipe_names


def lemmatize_tokens(text: str) -> list[str]:
    """Lower-cased lemmas with stop words and punctuation removed."""
    nlp, _ = get_nlp()
    if nlp is not None and "lemmatizer" in nlp.pipe_names:
        doc = nlp(text[:100_000])
        return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop and len(t) > 1]
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.]*", text.lower())
    return [t.strip(".") for t in tokens if t not in _STOPWORDS and len(t) > 1]


def key_phrases(text: str, top_n: int = 15) -> list[str]:
    """Most frequent noun phrases (spaCy noun chunks) or bigrams as a fallback."""
    nlp, _ = get_nlp()
    counter: Counter[str] = Counter()
    if nlp is not None and "parser" in nlp.pipe_names:
        doc = nlp(text[:100_000])
        for chunk in doc.noun_chunks:
            words = [t.lemma_.lower() for t in chunk if t.is_alpha and not t.is_stop]
            if 1 <= len(words) <= 4 and any(len(w) > 2 for w in words):
                counter[" ".join(words)] += 1
    else:
        tokens = lemmatize_tokens(text)
        for a, b in zip(tokens, tokens[1:]):
            counter[f"{a} {b}"] += 1
        for t in tokens:
            if len(t) > 3:
                counter[t] += 0.5
    return [phrase for phrase, _ in counter.most_common(top_n)]


def person_names(text: str) -> list[str]:
    """PERSON entities from the first part of a document (needs a trained model)."""
    nlp, _ = get_nlp()
    if nlp is None or "ner" not in nlp.pipe_names:
        return []
    doc = nlp(text[:600])
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
