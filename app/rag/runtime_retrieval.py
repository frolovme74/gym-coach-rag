"""BM25 с приоритетом практической карточки для явно базового вопроса."""

import re


class RuntimeRetrieval:
    def __init__(self, base, manifest):
        self.base = base
        self.rules = {
            m["doc_id"]: m.get("answer_patterns", [])
            for m in manifest
            if m.get("answer_patterns")
        }
        self.exclusions = {
            m["doc_id"]: m.get("answer_exclude_patterns", []) for m in manifest
        }

    def __getattr__(self, key):
        return getattr(self.base, key)

    def search(self, query, k=5):
        if type(k) is not int:
            raise TypeError("k должен быть целым числом")
        if k < 0:
            raise ValueError("k не может быть отрицательным")
        hits = self.base.search(query, k=self.base.n)
        preferred = self.preferred_ids(query)
        # Stable ordering; scores remain genuine BM25 scores, not probabilities.
        return sorted(hits, key=lambda h: h.doc_id not in preferred)[:k]

    def preferred_ids(self, query):
        text = query.casefold().replace("ё", "е")
        return {
            did
            for did, rules in self.rules.items()
            if any(re.search(rule, text) for rule in rules)
            and not any(re.search(rule, text) for rule in self.exclusions.get(did, []))
        }
