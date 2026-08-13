"""Person 1 definition-of-done: retrieve similar lessons for a new failure."""

from __future__ import annotations

import json
import sys

from pcb_memory.memory import retrieve_similar_lessons

QUERY = "Small open circuits have low recall and class imbalance"


def main() -> None:
    query = " ".join(sys.argv[1:]) or QUERY
    hits = retrieve_similar_lessons(query, k=5)
    print(json.dumps(hits, default=str, indent=2))


if __name__ == "__main__":
    main()
