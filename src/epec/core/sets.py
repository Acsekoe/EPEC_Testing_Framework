from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Sets:
    R: List[str]
    RR: List[Tuple[str, str]]   # directed arcs (e,r) with e!=r (trade arcs)
    RRx: List[Tuple[str, str]]  # all pairs (e,r) incl. diagonal (domestic flow)

def build_sets(regions: List[str]) -> Sets:
    rr = [(e, r) for e in regions for r in regions if e != r]
    rrx = [(e, r) for e in regions for r in regions]
    return Sets(R=regions, RR=rr, RRx=rrx)
