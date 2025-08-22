import re
import time
from typing import List

STOPWORDS = {
    "a","an","the","and","or","for","of","to","in","on","is","are","was","were","by"
}

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+"," ", s).strip()

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9$%.,:\-()\s]", " ", s)
    s = normalize_spaces(s)
    return s

def tokenize(s: str) -> List[str]:
    return [w for w in re.split(r"\W+", s.lower()) if w and w not in STOPWORDS]

# Guardrails
def is_in_scope_query(q: str) -> bool:
    ql = q.lower()
    bad = ["password", "ssn", "credit card", "bomb"]
    return not any(b in ql for b in bad)

# Simple confidence proxy
def softmax(xs):
    import math
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp(x-m) for x in xs]
    s = sum(ex) or 1.0
    return [e/s for e in ex]

class Timer:
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *args):
        self.dt = time.time() - self.t0
