"""Shared text vocabulary.

Lives here rather than in one consumer because both the FTS query builder and
the query planner need the same stopword opinion, and they must not drift apart.
"""

from __future__ import annotations

# Function words carry no retrieval signal. This matters more than it looks:
# FTS terms are OR-combined, so leaving "in", "is" or "the" in a query makes
# every memory match every question, which silently disables relevance gating.
STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also", "am",
        "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "cannot",
        "could", "did", "do", "does", "doing", "done", "down", "during", "each",
        "few", "for", "from", "further", "had", "has", "have", "having", "he",
        "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
        "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
        "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
        "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there", "these",
        "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
        "yourself", "yourselves",
        # Interrogative/instruction shells common in benchmark questions.
        "according", "answer", "based", "best", "choose", "correct", "correctly",
        "following", "given", "likely", "match", "matches", "option", "options",
        "please", "question", "select",
    }
)


def is_stopword(token: str) -> bool:
    return token.lower() in STOPWORDS
