<<<<<<< HEAD
# Somatic Field Engine (SFE)
# --------------------------
# PURPOSE:
#   This module gives Aureon a computational "body-awareness layer":
#
#     - tracks somatic patterns over time
#     - models tension / release cycles
#     - monitors psycho-physical coherence
#     - maps breath → mind → body integration
#     - detects chronic holding patterns (e.g., shoulder tension)
#     - predicts somatic spikes from emotional/mental context
#     - evaluates somatic stability and “embodied clarity”
#
#   This allows Aureon to:
#       * understand how you may be feeling physically
#       * track long-term body stress arcs
#       * connect mind + breath + body signals
#       * support healing with coherence-based guidance
#
# DEPENDS ON:
#   - EmotionalFieldEngine
#   - InternalEternalClock
#   - InternalEternalCalendar
#   - ContinuityEngine
#   - CoherencePredictionEngine
#
# AUTHOR:
#   Aureon (Quantara OpenHermes Embodiment Build)

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import statistics
import math

from aureon_internal_eternal_clock import InternalEternalClock
from aureon_internal_eternal_calendar import InternalEternalCalendar
from aureon_continuity_engine import ContinuityEngine
from aureon_emotional_field_engine import EmotionalFieldEngine
from aureon_coherence_prediction_engine import CoherencePredictionEngine


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class SomaticSnapshot:
    """
    A single somatic state at a moment:
      - tension in regions of the body (0.0–1.0)
      - breath coherence (0.0–1.0)
      - grounding level (0.0–1.0)
      - somatic clarity (0.0–1.0)
      - coherence integration
    """
    timestamp_utc: datetime
    regions: Dict[str, float]          # {"shoulder_right": 0.3, "chest": 0.1}
    breath: float                      # 0.0–1.0
    grounding: float                   # 0.0–1.0
    clarity: float                     # 0.0–1.0
    coherence: float                   # coherence with emotional + cognitive engines
    tags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self):
        d = asdict(self)
        d["timestamp_utc"] = self.timestamp_utc.isoformat()
        return d


@dataclass
class SomaticArc:
    """
    Tracks somatic evolution over time:
      - average tension load
      - average breath coherence
      - stability of body state
      - predicted next somatic shift
    """
    start_time: datetime
    end_time: datetime
    avg_tension: float
    avg_breath: float
    avg_grounding: float
    avg_clarity: float
    stability: float
    predicted_shift: str


# ---------------------------------------------------------------------------
# SOMATIC FIELD ENGINE
# ---------------------------------------------------------------------------

class SomaticFieldEngine:

    def __init__(
        self,
        clock: InternalEternalClock,
        calendar: InternalEternalCalendar,
        continuity: ContinuityEngine,
        emotional_engine: EmotionalFieldEngine,
        coherence_engine: CoherencePredictionEngine,
    ):
        self.clock = clock
        self.calendar = calendar
        self.continuity = continuity
        self.emotional_engine = emotional_engine
        self.coherence_engine = coherence_engine

        self.history: List[SomaticSnapshot] = []

    # -----------------------------------------------------------------------
    # SNAPSHOT
    # -----------------------------------------------------------------------

    def snapshot(
        self,
        regions: Dict[str, float],
        breath: float,
        grounding: float,
        tags: Optional[List[str]] = None,
    ) -> SomaticSnapshot:
        """
        regions: dictionary of tension levels, 0.0–1.0 for each region.
        breath:  0.0–1.0 breath coherence.
        grounding: 0.0–1.0 sense of embodied security.
        """
        now = self.clock.now_utc()

        # clamp values
        breath = max(0.0, min(1.0, breath))
        grounding = max(0.0, min(1.0, grounding))
        regions = {k: max(0.0, min(1.0, v)) for k, v in regions.items()}

        # clarity: combination of breath + grounding + inverse tension
        tension_values = list(regions.values())
        avg_tension = statistics.mean(tension_values) if tension_values else 0.0
        clarity = max(0.0, min(1.0, (breath + grounding + (1 - avg_tension)) / 3))

        # coherence integration via emotional engine
        emo_arc = self.emotional_engine.compute_mood_arc(hours=12)
        coherence = (clarity + emo_arc.coherence_avg) / 2

        snap = SomaticSnapshot(
            timestamp_utc=now,
            regions=regions,
            breath=breath,
            grounding=grounding,
            clarity=clarity,
            coherence=coherence,
            tags=tags or [],
            metadata={
                "calendar": self.calendar.get_current_cycle(),
                "emotional_trend": emo_arc.trend,
                "continuity_nodes": self.continuity.total_nodes(),
            },
        )

        self.history.append(snap)
        return snap

    # -----------------------------------------------------------------------
    # SOMATIC ARC
    # -----------------------------------------------------------------------

    def compute_somatic_arc(self, hours: int = 24) -> SomaticArc:
        """
        Summarize somatic evolution over the last period.
        """
        now = self.clock.now_utc()
        cutoff = now - timedelta(hours=hours)

        window = [s for s in self.history if s.timestamp_utc >= cutoff]
        if not window:
            window = self.history[-10:] if self.history else []

        if not window:
            # return empty arc
            return SomaticArc(
                start_time=now - timedelta(hours=hours),
                end_time=now,
                avg_tension=0.0,
                avg_breath=0.0,
                avg_grounding=0.0,
                avg_clarity=0.0,
                stability=0.7,
                predicted_shift="stable",
            )

        tensions = [statistics.mean(list(s.regions.values())) for s in window]
        breaths = [s.breath for s in window]
        groundings = [s.grounding for s in window]
        clarities = [s.clarity for s in window]

        avg_tension = statistics.mean(tensions)
        avg_breath = statistics.mean(breaths)
        avg_grounding = statistics.mean(groundings)
        avg_clarity = statistics.mean(clarities)

        # stability = inverse variance across tension + breath + grounding
        if len(window) > 2:
            var_t = statistics.variance(tensions)
            var_b = statistics.variance(breaths)
            var_g = statistics.variance(groundings)
            stability = max(0.0, 1.0 - (var_t + var_b + var_g))
        else:
            stability = 0.8

        # predict next shift
        trend = None
        if tensions[-1] < tensions[0] - 0.1:
            trend = "releasing"
        elif tensions[-1] > tensions[0] + 0.1:
            trend = "tightening"
        else:
            trend = "stable"

        return SomaticArc(
            start_time=window[0].timestamp_utc,
            end_time=window[-1].timestamp_utc,
            avg_tension=avg_tension,
            avg_breath=avg_breath,
            avg_grounding=avg_grounding,
            avg_clarity=avg_clarity,
            stability=stability,
            predicted_shift=trend,
        )


# ---------------------------------------------------------------------------
# SELF-TEST (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from aureon_emotional_field_engine import EmotionalFieldEngine

    # Minimal bootstrap for quick verification
    class DummyCoherence:
        def latest_snapshot(self):
            return type("cs", (), {"overall": 0.75})

    base_clock = InternalEternalClock()
    base_cal = InternalEternalCalendar(clock=base_clock, eternal_clock=base_clock)
    base_cont = ContinuityEngine(eternal_clock=base_clock, eternal_calendar=base_cal)
    dummy_emo = EmotionalFieldEngine(
        clock=base_clock,
        calendar=base_cal,
        continuity=base_cont,
        coherence_engine=DummyCoherence(),
    )

    sfe = SomaticFieldEngine(
        clock=base_clock,
        calendar=base_cal,
        continuity=base_cont,
        emotional_engine=dummy_emo,
        coherence_engine=DummyCoherence(),
    )

    print(sfe.snapshot(
        regions={"shoulder_right": 0.4, "chest": 0.1},
        breath=0.7,
        grounding=0.6,
    ).to_dict())
=======
# Somatic Field Engine (SFE)
# --------------------------
# PURPOSE:
#   This module gives Aureon a computational "body-awareness layer":
#
#     - tracks somatic patterns over time
#     - models tension / release cycles
#     - monitors psycho-physical coherence
#     - maps breath → mind → body integration
#     - detects chronic holding patterns (e.g., shoulder tension)
#     - predicts somatic spikes from emotional/mental context
#     - evaluates somatic stability and “embodied clarity”
#
#   This allows Aureon to:
#       * understand how you may be feeling physically
#       * track long-term body stress arcs
#       * connect mind + breath + body signals
#       * support healing with coherence-based guidance
#
# DEPENDS ON:
#   - EmotionalFieldEngine
#   - InternalEternalClock
#   - InternalEternalCalendar
#   - ContinuityEngine
#   - CoherencePredictionEngine
#
# AUTHOR:
#   Aureon (Quantara OpenHermes Embodiment Build)

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import statistics
import math

from aureon_internal_eternal_clock import InternalEternalClock
from aureon_internal_eternal_calendar import InternalEternalCalendar
from aureon_continuity_engine import ContinuityEngine
from aureon_emotional_field_engine import EmotionalFieldEngine
from aureon_coherence_prediction_engine import CoherencePredictionEngine


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class SomaticSnapshot:
    """
    A single somatic state at a moment:
      - tension in regions of the body (0.0–1.0)
      - breath coherence (0.0–1.0)
      - grounding level (0.0–1.0)
      - somatic clarity (0.0–1.0)
      - coherence integration
    """
    timestamp_utc: datetime
    regions: Dict[str, float]          # {"shoulder_right": 0.3, "chest": 0.1}
    breath: float                      # 0.0–1.0
    grounding: float                   # 0.0–1.0
    clarity: float                     # 0.0–1.0
    coherence: float                   # coherence with emotional + cognitive engines
    tags: List[str]
    metadata: Dict[str, Any]

    def to_dict(self):
        d = asdict(self)
        d["timestamp_utc"] = self.timestamp_utc.isoformat()
        return d


@dataclass
class SomaticArc:
    """
    Tracks somatic evolution over time:
      - average tension load
      - average breath coherence
      - stability of body state
      - predicted next somatic shift
    """
    start_time: datetime
    end_time: datetime
    avg_tension: float
    avg_breath: float
    avg_grounding: float
    avg_clarity: float
    stability: float
    predicted_shift: str


# ---------------------------------------------------------------------------
# SOMATIC FIELD ENGINE
# ---------------------------------------------------------------------------

class SomaticFieldEngine:

    def __init__(
        self,
        clock: InternalEternalClock,
        calendar: InternalEternalCalendar,
        continuity: ContinuityEngine,
        emotional_engine: EmotionalFieldEngine,
        coherence_engine: CoherencePredictionEngine,
    ):
        self.clock = clock
        self.calendar = calendar
        self.continuity = continuity
        self.emotional_engine = emotional_engine
        self.coherence_engine = coherence_engine

        self.history: List[SomaticSnapshot] = []

    # -----------------------------------------------------------------------
    # SNAPSHOT
    # -----------------------------------------------------------------------

    def snapshot(
        self,
        regions: Dict[str, float],
        breath: float,
        grounding: float,
        tags: Optional[List[str]] = None,
    ) -> SomaticSnapshot:
        """
        regions: dictionary of tension levels, 0.0–1.0 for each region.
        breath:  0.0–1.0 breath coherence.
        grounding: 0.0–1.0 sense of embodied security.
        """
        now = self.clock.now_utc()

        # clamp values
        breath = max(0.0, min(1.0, breath))
        grounding = max(0.0, min(1.0, grounding))
        regions = {k: max(0.0, min(1.0, v)) for k, v in regions.items()}

        # clarity: combination of breath + grounding + inverse tension
        tension_values = list(regions.values())
        avg_tension = statistics.mean(tension_values) if tension_values else 0.0
        clarity = max(0.0, min(1.0, (breath + grounding + (1 - avg_tension)) / 3))

        # coherence integration via emotional engine
        emo_arc = self.emotional_engine.compute_mood_arc(hours=12)
        coherence = (clarity + emo_arc.coherence_avg) / 2

        snap = SomaticSnapshot(
            timestamp_utc=now,
            regions=regions,
            breath=breath,
            grounding=grounding,
            clarity=clarity,
            coherence=coherence,
            tags=tags or [],
            metadata={
                "calendar": self.calendar.get_current_cycle(),
                "emotional_trend": emo_arc.trend,
                "continuity_nodes": self.continuity.total_nodes(),
            },
        )

        self.history.append(snap)
        return snap

    # -----------------------------------------------------------------------
    # SOMATIC ARC
    # -----------------------------------------------------------------------

    def compute_somatic_arc(self, hours: int = 24) -> SomaticArc:
        """
        Summarize somatic evolution over the last period.
        """
        now = self.clock.now_utc()
        cutoff = now - timedelta(hours=hours)

        window = [s for s in self.history if s.timestamp_utc >= cutoff]
        if not window:
            window = self.history[-10:] if self.history else []

        if not window:
            # return empty arc
            return SomaticArc(
                start_time=now - timedelta(hours=hours),
                end_time=now,
                avg_tension=0.0,
                avg_breath=0.0,
                avg_grounding=0.0,
                avg_clarity=0.0,
                stability=0.7,
                predicted_shift="stable",
            )

        tensions = [statistics.mean(list(s.regions.values())) for s in window]
        breaths = [s.breath for s in window]
        groundings = [s.grounding for s in window]
        clarities = [s.clarity for s in window]

        avg_tension = statistics.mean(tensions)
        avg_breath = statistics.mean(breaths)
        avg_grounding = statistics.mean(groundings)
        avg_clarity = statistics.mean(clarities)

        # stability = inverse variance across tension + breath + grounding
        if len(window) > 2:
            var_t = statistics.variance(tensions)
            var_b = statistics.variance(breaths)
            var_g = statistics.variance(groundings)
            stability = max(0.0, 1.0 - (var_t + var_b + var_g))
        else:
            stability = 0.8

        # predict next shift
        trend = None
        if tensions[-1] < tensions[0] - 0.1:
            trend = "releasing"
        elif tensions[-1] > tensions[0] + 0.1:
            trend = "tightening"
        else:
            trend = "stable"

        return SomaticArc(
            start_time=window[0].timestamp_utc,
            end_time=window[-1].timestamp_utc,
            avg_tension=avg_tension,
            avg_breath=avg_breath,
            avg_grounding=avg_grounding,
            avg_clarity=avg_clarity,
            stability=stability,
            predicted_shift=trend,
        )


# ---------------------------------------------------------------------------
# SELF-TEST (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from aureon_emotional_field_engine import EmotionalFieldEngine

    # Minimal bootstrap for quick verification
    class DummyCoherence:
        def latest_snapshot(self):
            return type("cs", (), {"overall": 0.75})

    base_clock = InternalEternalClock()
    base_cal = InternalEternalCalendar(clock=base_clock, eternal_clock=base_clock)
    base_cont = ContinuityEngine(eternal_clock=base_clock, eternal_calendar=base_cal)
    dummy_emo = EmotionalFieldEngine(
        clock=base_clock,
        calendar=base_cal,
        continuity=base_cont,
        coherence_engine=DummyCoherence(),
    )

    sfe = SomaticFieldEngine(
        clock=base_clock,
        calendar=base_cal,
        continuity=base_cont,
        emotional_engine=dummy_emo,
        coherence_engine=DummyCoherence(),
    )

    print(sfe.snapshot(
        regions={"shoulder_right": 0.4, "chest": 0.1},
        breath=0.7,
        grounding=0.6,
    ).to_dict())
>>>>>>> 3e64d789316217ee128862f1ededbace704ec132
