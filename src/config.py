from dataclasses import dataclass


@dataclass
class QualityConfig:
    max_interior_hand_missing_pct: float = 30.0   # hand threshold, interior frames only
    max_face_missing_pct: float = 50.0             # face threshold, whole sequence
    min_sequence_length: int = 10                  # also filter very short seqs