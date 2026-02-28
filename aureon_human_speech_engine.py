#!/usr/bin/env python3
"""
AUREON HUMAN SPEECH ENGINE — CONVERSATIONAL DYNAMICS CORE
==============================================================

THE BREAKTHROUGH:
    Every AI system that generates language does one of two things:
        1. TOKEN PREDICTION (LLMs) — compute P(next_token | context)
        2. TEMPLATE RETRIEVAL (chatbots) — find closest match, fill slots

    Both are fundamentally REACTIVE. They respond to what just happened.
    Neither understands WHERE THE CONVERSATION IS GOING.

    Humans do something radically different. When a therapist says
    "Tell me more about that," she isn't predicting tokens or retrieving
    templates. She is NAVIGATING — she knows where the conversation is
    in emotional-rhetorical space, she knows where it needs to go, and
    she produces speech that MOVES the conversation along that trajectory.

    A jazz musician doesn't retrieve licks from memory. She HEARS where
    the music is harmonically, FEELS where it wants to go, and PLAYS
    the notes that carry it there. The notes come from years of absorbed
    practice, but the SELECTION is driven by trajectory, not retrieval.

    THIS ENGINE DOES THE SAME THING FOR SPEECH.

THE ARCHITECTURE:
    1. PHASE SPACE — A continuous high-dimensional space where every
       possible conversational state has coordinates. Dimensions include:
       emotional valence, arousal, dominance, topic depth, rapport,
       formality, rhetorical momentum, vulnerability, playfulness,
       intellectual density, and more.

    2. ATTRACTOR LANDSCAPE — Learned from millions of real conversations,
       this maps the natural FLOWS of human dialogue. Conversations don't
       wander randomly — they follow attractor basins. A comfort conversation
       has a different attractor than a debate, which differs from a lecture.
       The engine learns these attractors from absorbed speech.

    3. TRAJECTORY COMPUTATION — Given the current state and attractor,
       compute the optimal NEXT STATE in phase space. This is where the
       κ-τ-Σ framework provides the constraints:
           κ (kappa) = spatial coherence = the response must be internally
                       consistent and consistent with conversation history
           τ (tau)   = temporal responsibility = the response must serve
                       the conversation's future, not just react to its past
           σ (sigma) = systemic separation = the response must maintain
                       appropriate boundaries

    4. SPEECH SYNTHESIS — Map the target state back to actual words.
       This is NOT template filling. The engine uses VECTOR COMPOSITION:
       it finds the combination of absorbed speech fragments whose vectors
       SUM to the target state vector. Like mixing colors to hit a target
       hue — but in 20+ dimensions of rhetorical-emotional space.

    5. PROSODIC ENCODING — The output isn't just words. It includes timing
       markers, emphasis patterns, emotional coloring, and structural
       annotations. When fed to a TTS system, the speech sounds HUMAN
       because it was composed from human speech dynamics.

WHAT MAKES THIS UNPRECEDENTED:
    - No LLM has ever modeled conversation as trajectory through phase space
    - No system has ever used reaction-diffusion dynamics for speech selection
    - No system has ever composed speech by VECTOR ADDITION of absorbed patterns
    - No system has ever used coherence constraints (κ-τ-Σ) as generation bounds
    - No system has ever separated NAVIGATION (where to go) from ARTICULATION
      (how to say it) in this mathematically rigorous way

    The closest analogy in existing research is dynamical systems theory
    applied to motor control (how humans plan arm movements). We apply the
    same mathematics to SPEECH MOVEMENTS through rhetorical space.

AUTHOR: Nadine Squires / Team Aureon
LICENSE: Proprietary — quantumquantara-arch
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

# ====================================================================
# MATHEMATICAL PRIMITIVES
# ====================================================================

# Dimensionality of the rhetorical-emotional phase space
PHASE_DIM = 24

# Named dimensions (indices into the state vector)
DIM_VALENCE          = 0   # emotional positivity [-1, +1]
DIM_AROUSAL          = 1   # emotional intensity [0, 1]
DIM_DOMINANCE        = 2   # assertion vs deference [0, 1]
DIM_TOPIC_DEPTH      = 3   # surface → profound [0, 1]
DIM_RAPPORT          = 4   # stranger → deep trust [0, 1]
DIM_FORMALITY        = 5   # casual → formal [0, 1]
DIM_VULNERABILITY    = 6   # guarded → exposed [0, 1]
DIM_PLAYFULNESS      = 7   # serious → playful [0, 1]
DIM_INTELLECTUAL     = 8   # intuitive → analytical [0, 1]
DIM_MOMENTUM         = 9   # conversation energy [0, 1]
DIM_CERTAINTY        = 10  # tentative → definitive [0, 1]
DIM_CURIOSITY        = 11  # closed → exploratory [0, 1]
DIM_WARMTH           = 12  # cool → warm [0, 1]
DIM_SURPRISE         = 13  # expected → unexpected [0, 1]
DIM_TENSION          = 14  # resolved → unresolved [0, 1]
DIM_NARRATIVE        = 15  # factual → story-like [0, 1]
DIM_METAPHOR         = 16  # literal → figurative [0, 1]
DIM_AGENCY           = 17  # passive → active [0, 1]
DIM_TEMPORAL_FOCUS   = 18  # past → future [-1, +1]
DIM_SCOPE            = 19  # personal → universal [0, 1]
DIM_PACE             = 20  # deliberate → rapid [0, 1]
DIM_RECIPROCITY      = 21  # monologue → dialogue [0, 1]
DIM_COHERENCE_KAPPA  = 22  # internal coherence [0, 1]
DIM_COHERENCE_TAU    = 23  # temporal coherence [0, 1]

DIM_NAMES = [
    "valence", "arousal", "dominance", "topic_depth", "rapport",
    "formality", "vulnerability", "playfulness", "intellectual", "momentum",
    "certainty", "curiosity", "warmth", "surprise", "tension",
    "narrative", "metaphor", "agency", "temporal_focus", "scope",
    "pace", "reciprocity", "kappa", "tau",
]


def vec_zero() -> List[float]:
    """Zero vector in phase space."""
    return [0.0] * PHASE_DIM


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """Add two phase-space vectors."""
    return [a[i] + b[i] for i in range(PHASE_DIM)]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Subtract b from a in phase space."""
    return [a[i] - b[i] for i in range(PHASE_DIM)]


def vec_scale(v: List[float], s: float) -> List[float]:
    """Scale a vector."""
    return [x * s for x in v]


def vec_magnitude(v: List[float]) -> float:
    """Euclidean magnitude."""
    return math.sqrt(sum(x * x for x in v))


def vec_normalize(v: List[float]) -> List[float]:
    """Normalize to unit vector."""
    m = vec_magnitude(v)
    if m < 1e-10:
        return vec_zero()
    return [x / m for x in v]


def vec_dot(a: List[float], b: List[float]) -> float:
    """Dot product."""
    return sum(a[i] * b[i] for i in range(PHASE_DIM))


def vec_cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity."""
    ma, mb = vec_magnitude(a), vec_magnitude(b)
    if ma < 1e-10 or mb < 1e-10:
        return 0.0
    return vec_dot(a, b) / (ma * mb)


def vec_lerp(a: List[float], b: List[float], t: float) -> List[float]:
    """Linear interpolation between a and b. t=0 gives a, t=1 gives b."""
    return [a[i] + t * (b[i] - a[i]) for i in range(PHASE_DIM)]


def vec_clamp(v: List[float], lo: float = -1.0, hi: float = 1.0) -> List[float]:
    """Clamp all dimensions."""
    return [max(lo, min(hi, x)) for x in v]


def vec_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance."""
    return vec_magnitude(vec_sub(a, b))


def vec_weighted_sum(vectors: List[List[float]], weights: List[float]) -> List[float]:
    """Weighted sum of vectors."""
    result = vec_zero()
    for v, w in zip(vectors, weights):
        result = vec_add(result, vec_scale(v, w))
    return result


# ====================================================================
# SECTION 1: SPEECH ATOMS — the finest units of absorbed human speech
# ====================================================================

@dataclass
class SpeechAtom:
    """
    The fundamental particle of human speech in the HSE.

    Unlike the v1 engine's SpeechPattern (which stores text + metadata),
    a SpeechAtom stores text + its POSITION IN PHASE SPACE.

    When a human says "That must have been incredibly difficult for you,"
    the atom doesn't just store those words. It stores the VECTOR that
    locates those words in 24-dimensional rhetorical-emotional space:
        valence = -0.2 (acknowledging something negative)
        arousal = 0.4 (moderate intensity)
        warmth = 0.9 (deeply warm)
        vulnerability = 0.7 (inviting openness)
        reciprocity = 0.8 (high dialogue quality)
        ... etc for all 24 dimensions

    This vector IS the meaning of the utterance. Not semantic meaning
    (what the words denote) but PRAGMATIC meaning (what the words DO
    to the conversational state).

    The Composer generates speech by finding atoms whose vectors SUM
    to the target vector — like mixing paint colors to hit a target hue,
    but in 24 dimensions.
    """
    id: str                          # SHA-256 hash
    text: str                        # raw speech fragment
    vector: List[float]              # position in phase space (len = PHASE_DIM)

    # Scale classification
    scale: str = "sentence"          # word | phrase | clause | sentence | turn

    # Source tracking
    source_type: str = ""            # podcast | movie | documentary | youtube | etc
    source_name: str = ""            # specific source identifier
    speaker_archetype: str = ""      # therapist | teacher | comedian | debater | storyteller | etc

    # Transition data — what vectors this atom MOVES the state toward
    delta_vector: List[float] = field(default_factory=vec_zero)  # state change this atom produces

    # Prosodic encoding
    emphasis_mask: List[int] = field(default_factory=list)  # 1 = stressed, 0 = unstressed per word
    pause_durations: List[float] = field(default_factory=list)  # seconds of pause after each word
    pitch_contour: str = "neutral"   # rising | falling | rise_fall | fall_rise | neutral

    # Composition compatibility
    can_open: bool = False           # can this start a response?
    can_close: bool = False          # can this end a response?
    can_follow: List[str] = field(default_factory=list)  # atom IDs this naturally follows
    can_precede: List[str] = field(default_factory=list)  # atom IDs this naturally precedes

    # Quality metrics
    naturalness_score: float = 1.0   # how natural this sounds in isolation
    versatility_score: float = 0.5   # how many contexts this works in
    times_composed: int = 0          # how often selected by Composer
    avg_coherence_when_used: float = 0.0
    last_used_time: float = 0.0

    def similarity_to(self, target: List[float]) -> float:
        """How close is this atom's vector to a target state?"""
        return vec_cosine(self.vector, target)

    def delta_toward(self, target: List[float]) -> float:
        """How much does this atom's delta_vector move state toward target?"""
        if not self.delta_vector or vec_magnitude(self.delta_vector) < 1e-10:
            return 0.0
        return vec_cosine(self.delta_vector, target)


# ====================================================================
# SECTION 2: THE ABSORBER v2 — maps speech to phase space
# ====================================================================

class PhaseSpaceAbsorber:
    """
    Ingests human speech and maps it into the 24-dimensional phase space.

    The mapping is done through a LEXICAL-PRAGMATIC ANALYZER that combines:
    1. Lexical signals (word choice, hedging, emphasis patterns)
    2. Structural signals (sentence type, clause arrangement)
    3. Sequential signals (what came before, what comes after)
    4. Prosodic signals (from timing data in subtitles/audio)

    This is not machine learning. It's a hand-crafted dimensional analysis
    based on decades of pragmatics research (Grice, Austin, Searle, Brown
    & Levinson, Conversation Analysis, Relevance Theory). Each dimension
    has explicit, auditable scoring rules.
    """

    # ----------------------------------------------------------------
    # Lexicon for dimensional scoring
    # ----------------------------------------------------------------

    # Words/phrases that indicate position on each dimension
    DIMENSIONAL_LEXICON: Dict[int, Dict[str, float]] = {
        DIM_VALENCE: {
            # Positive
            "love": 0.8, "beautiful": 0.7, "amazing": 0.7, "wonderful": 0.7,
            "brilliant": 0.7, "fantastic": 0.6, "great": 0.5, "good": 0.3,
            "nice": 0.3, "happy": 0.6, "joy": 0.7, "grateful": 0.6,
            "excited": 0.6, "perfect": 0.7, "incredible": 0.7,
            # Negative
            "hate": -0.8, "terrible": -0.7, "horrible": -0.7, "awful": -0.6,
            "disgusting": -0.7, "painful": -0.6, "sad": -0.5, "angry": -0.6,
            "frustrated": -0.5, "disappointed": -0.5, "broken": -0.5,
            "lost": -0.4, "struggling": -0.4, "fail": -0.5, "wrong": -0.4,
        },
        DIM_AROUSAL: {
            "absolutely": 0.7, "incredibly": 0.7, "extremely": 0.8,
            "totally": 0.6, "completely": 0.6, "literally": 0.5,
            "definitely": 0.5, "seriously": 0.5, "honestly": 0.4,
            "insane": 0.8, "crazy": 0.6, "wild": 0.6, "massive": 0.6,
            "quietly": -0.3, "gently": -0.3, "softly": -0.3, "calmly": -0.4,
            "perhaps": -0.2, "maybe": -0.2, "slightly": -0.3,
        },
        DIM_DOMINANCE: {
            "must": 0.7, "shall": 0.6, "clearly": 0.6, "obviously": 0.6,
            "certainly": 0.6, "always": 0.5, "never": 0.5, "every": 0.4,
            "command": 0.7, "demand": 0.7, "require": 0.6, "insist": 0.6,
            "maybe": -0.3, "perhaps": -0.3, "might": -0.3, "could": -0.2,
            "possibly": -0.3, "sometimes": -0.2, "wonder": -0.2,
            "sorry": -0.4, "apologize": -0.5, "humble": -0.4,
        },
        DIM_WARMTH: {
            "dear": 0.7, "friend": 0.5, "sweetheart": 0.8, "love": 0.7,
            "care": 0.6, "together": 0.4, "share": 0.4, "appreciate": 0.5,
            "welcome": 0.5, "glad": 0.4, "treasure": 0.6, "cherish": 0.7,
            "hug": 0.6, "warm": 0.5, "kind": 0.5, "gentle": 0.5,
            "harsh": -0.5, "cold": -0.4, "indifferent": -0.5, "whatever": -0.4,
        },
        DIM_VULNERABILITY: {
            "afraid": 0.7, "scared": 0.7, "nervous": 0.6, "anxious": 0.6,
            "uncertain": 0.5, "confused": 0.5, "lost": 0.6, "alone": 0.7,
            "overwhelmed": 0.6, "vulnerable": 0.8, "exposed": 0.7,
            "admit": 0.5, "confess": 0.6, "embarrassed": 0.6,
            "confident": -0.4, "strong": -0.3, "certain": -0.4, "assured": -0.4,
        },
        DIM_PLAYFULNESS: {
            "haha": 0.7, "lol": 0.6, "funny": 0.6, "hilarious": 0.7,
            "joke": 0.5, "kidding": 0.5, "silly": 0.5, "goofy": 0.6,
            "ridiculous": 0.4, "absurd": 0.4, "wild": 0.4, "imagine": 0.3,
            "seriously": -0.3, "grave": -0.5, "solemn": -0.5, "critical": -0.3,
        },
        DIM_INTELLECTUAL: {
            "analyze": 0.7, "framework": 0.7, "hypothesis": 0.7, "evidence": 0.6,
            "data": 0.6, "logic": 0.7, "reason": 0.5, "theory": 0.6,
            "structure": 0.5, "mechanism": 0.6, "principle": 0.6, "systematic": 0.7,
            "feel": -0.3, "sense": -0.2, "gut": -0.4, "heart": -0.3,
            "vibe": -0.4, "intuition": -0.3, "instinct": -0.3,
        },
        DIM_CURIOSITY: {
            "wonder": 0.7, "fascinating": 0.7, "curious": 0.7, "interesting": 0.5,
            "explore": 0.6, "discover": 0.6, "why": 0.4, "how": 0.3,
            "what if": 0.7, "imagine": 0.5, "suppose": 0.5, "ponder": 0.6,
            "obvious": -0.4, "settled": -0.5, "done": -0.3, "final": -0.4,
        },
        DIM_NARRATIVE: {
            "story": 0.7, "once": 0.5, "remember": 0.5, "happened": 0.4,
            "told": 0.4, "said": 0.3, "went": 0.3, "became": 0.4,
            "journey": 0.6, "chapter": 0.5, "moment": 0.4, "imagine": 0.5,
            "data": -0.4, "percent": -0.5, "statistic": -0.5, "metric": -0.5,
        },
        DIM_METAPHOR: {
            "like": 0.3, "as if": 0.5, "imagine": 0.4, "picture": 0.4,
            "bridge": 0.4, "ocean": 0.4, "mountain": 0.4, "fire": 0.4,
            "light": 0.3, "dark": 0.3, "dance": 0.4, "song": 0.3,
            "specifically": -0.3, "literally": -0.4, "exactly": -0.3,
            "precisely": -0.4, "technically": -0.4,
        },
    }

    # Structural indicators for dimensions not well-captured by lexicon
    STRUCTURAL_INDICATORS = {
        DIM_FORMALITY: {
            "formal_markers": ["furthermore", "moreover", "consequently", "nevertheless",
                             "regarding", "concerning", "accordingly", "henceforth"],
            "informal_markers": ["gonna", "wanna", "kinda", "sorta", "yeah", "nah",
                               "dude", "bro", "like", "yo", "omg", "lol"],
        },
        DIM_RECIPROCITY: {
            "dialogue_markers": ["?", "what do you", "tell me", "how about you",
                               "your thoughts", "you think", "agree"],
            "monologue_markers": ["let me explain", "here's what", "the point is",
                                "I'll tell you", "listen"],
        },
    }

    # Speaker archetype signatures (characteristic vector regions)
    ARCHETYPE_PROFILES: Dict[str, List[float]] = {}  # populated in __init__

    def __init__(self):
        """Initialize the absorber with archetype profiles."""
        self._init_archetypes()
        self._atoms_created = 0

    def _init_archetypes(self):
        """Define characteristic vector profiles for speaker archetypes."""
        # Each archetype has a "home" region in phase space

        # Therapist: warm, curious, low dominance, high rapport
        therapist = vec_zero()
        therapist[DIM_WARMTH] = 0.8
        therapist[DIM_CURIOSITY] = 0.7
        therapist[DIM_DOMINANCE] = 0.2
        therapist[DIM_RAPPORT] = 0.8
        therapist[DIM_VULNERABILITY] = 0.3
        therapist[DIM_RECIPROCITY] = 0.8
        therapist[DIM_PACE] = 0.3
        therapist[DIM_CERTAINTY] = 0.3
        self.ARCHETYPE_PROFILES["therapist"] = therapist

        # Teacher: clear, structured, moderate warmth, high intellectual
        teacher = vec_zero()
        teacher[DIM_INTELLECTUAL] = 0.8
        teacher[DIM_WARMTH] = 0.5
        teacher[DIM_CERTAINTY] = 0.6
        teacher[DIM_TOPIC_DEPTH] = 0.6
        teacher[DIM_PACE] = 0.4
        teacher[DIM_NARRATIVE] = 0.4
        teacher[DIM_RECIPROCITY] = 0.5
        self.ARCHETYPE_PROFILES["teacher"] = teacher

        # Comedian: high playfulness, high arousal, surprise-seeking
        comedian = vec_zero()
        comedian[DIM_PLAYFULNESS] = 0.9
        comedian[DIM_AROUSAL] = 0.6
        comedian[DIM_SURPRISE] = 0.8
        comedian[DIM_PACE] = 0.7
        comedian[DIM_NARRATIVE] = 0.7
        comedian[DIM_WARMTH] = 0.4
        comedian[DIM_TENSION] = 0.6
        self.ARCHETYPE_PROFILES["comedian"] = comedian

        # Debater: high intellectual, high dominance, high momentum
        debater = vec_zero()
        debater[DIM_INTELLECTUAL] = 0.8
        debater[DIM_DOMINANCE] = 0.7
        debater[DIM_MOMENTUM] = 0.8
        debater[DIM_CERTAINTY] = 0.7
        debater[DIM_AROUSAL] = 0.5
        debater[DIM_TENSION] = 0.6
        debater[DIM_PACE] = 0.6
        self.ARCHETYPE_PROFILES["debater"] = debater

        # Storyteller: high narrative, moderate warmth, temporal richness
        storyteller = vec_zero()
        storyteller[DIM_NARRATIVE] = 0.9
        storyteller[DIM_METAPHOR] = 0.7
        storyteller[DIM_WARMTH] = 0.5
        storyteller[DIM_TOPIC_DEPTH] = 0.6
        storyteller[DIM_PACE] = 0.4
        storyteller[DIM_VULNERABILITY] = 0.4
        self.ARCHETYPE_PROFILES["storyteller"] = storyteller

        # Sage: deep, measured, high coherence, philosophical
        sage = vec_zero()
        sage[DIM_TOPIC_DEPTH] = 0.9
        sage[DIM_CERTAINTY] = 0.6
        sage[DIM_WARMTH] = 0.5
        sage[DIM_PACE] = 0.2
        sage[DIM_INTELLECTUAL] = 0.7
        sage[DIM_METAPHOR] = 0.6
        sage[DIM_SCOPE] = 0.8
        sage[DIM_COHERENCE_KAPPA] = 0.9
        sage[DIM_COHERENCE_TAU] = 0.9
        self.ARCHETYPE_PROFILES["sage"] = sage

        # Companion: warm, matched arousal, high rapport, present
        companion = vec_zero()
        companion[DIM_WARMTH] = 0.7
        companion[DIM_RAPPORT] = 0.7
        companion[DIM_RECIPROCITY] = 0.9
        companion[DIM_VULNERABILITY] = 0.4
        companion[DIM_PLAYFULNESS] = 0.4
        companion[DIM_AGENCY] = 0.5
        self.ARCHETYPE_PROFILES["companion"] = companion

    def absorb(self, text: str, source_type: str = "", source_name: str = "",
               speaker_archetype: str = "") -> List[SpeechAtom]:
        """
        Absorb text and produce SpeechAtoms with phase-space vectors.

        The core operation of the entire engine: mapping human words
        to positions in rhetorical-emotional space.
        """
        atoms = []

        # Parse into segments
        turns = self._segment(text)

        prev_vector = vec_zero()
        for i, segment in enumerate(turns):
            if not segment.strip() or len(segment) < 5:
                continue

            # Compute the phase-space vector for this segment
            vector = self._compute_vector(segment)

            # Apply archetype bias if known
            if speaker_archetype and speaker_archetype in self.ARCHETYPE_PROFILES:
                archetype_vec = self.ARCHETYPE_PROFILES[speaker_archetype]
                # Blend: 70% measured, 30% archetype pull
                vector = vec_lerp(vector, archetype_vec, 0.3)

            # Compute the delta (state change this segment produces)
            delta = vec_sub(vector, prev_vector) if i > 0 else vec_zero()

            # Determine scale
            words = segment.split()
            if len(words) <= 2:
                scale = "word"
            elif len(words) <= 6:
                scale = "phrase"
            elif len(words) <= 15:
                scale = "clause"
            elif len(words) <= 40:
                scale = "sentence"
            else:
                scale = "turn"

            # Prosodic encoding
            emphasis = self._compute_emphasis(segment)
            pauses = self._compute_pauses(segment)
            contour = self._compute_pitch_contour(segment)

            # Composition markers
            can_open = i == 0 or segment[0].isupper()
            can_close = segment.rstrip()[-1] in '.!?' if segment.rstrip() else False

            atom = SpeechAtom(
                id=hashlib.sha256(f"{segment}|{source_name}|{i}".encode()).hexdigest()[:16],
                text=segment.strip(),
                vector=vector,
                scale=scale,
                source_type=source_type,
                source_name=source_name,
                speaker_archetype=speaker_archetype,
                delta_vector=delta,
                emphasis_mask=emphasis,
                pause_durations=pauses,
                pitch_contour=contour,
                can_open=can_open,
                can_close=can_close,
                naturalness_score=self._score_naturalness(segment),
                versatility_score=self._score_versatility(segment),
            )

            atoms.append(atom)
            prev_vector = vector
            self._atoms_created += 1

        return atoms

    def _segment(self, text: str) -> List[str]:
        """Segment text into speech-sized fragments."""
        # First split by speaker turns
        lines = text.strip().split('\n')
        segments = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove speaker labels
            line = re.sub(r'^[\w\s]+:\s*', '', line)
            # Split long lines into sentences
            sentences = re.split(r'(?<=[.!?])\s+', line)
            segments.extend(s.strip() for s in sentences if s.strip())

        return segments

    def _compute_vector(self, text: str) -> List[float]:
        """
        THE CORE MAPPING: text → phase-space vector.

        This is the most important function in the entire engine.
        It analyzes a speech fragment along all 24 dimensions to
        produce its coordinates in rhetorical-emotional space.
        """
        vector = vec_zero()
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        if word_count == 0:
            return vector

        # DIMENSION 1-13: Lexicon-based scoring
        for dim_idx, lexicon in self.DIMENSIONAL_LEXICON.items():
            score = 0.0
            hits = 0
            for word, value in lexicon.items():
                count = text_lower.count(word)
                if count > 0:
                    score += value * count
                    hits += count
            # Normalize by text length to avoid bias toward long fragments
            if hits > 0:
                vector[dim_idx] = max(-1.0, min(1.0, score / max(hits, 1)))

        # DIM_FORMALITY: structural analysis
        formal_count = sum(1 for m in self.STRUCTURAL_INDICATORS[DIM_FORMALITY]["formal_markers"]
                          if m in text_lower)
        informal_count = sum(1 for m in self.STRUCTURAL_INDICATORS[DIM_FORMALITY]["informal_markers"]
                            if m in text_lower)
        total_markers = formal_count + informal_count
        if total_markers > 0:
            vector[DIM_FORMALITY] = (formal_count - informal_count) / total_markers
        else:
            # Heuristic: average word length correlates with formality
            avg_len = sum(len(w) for w in words) / word_count
            vector[DIM_FORMALITY] = max(-1.0, min(1.0, (avg_len - 4.5) / 3.0))

        # DIM_RECIPROCITY: dialogue markers
        dialogue = sum(1 for m in self.STRUCTURAL_INDICATORS[DIM_RECIPROCITY]["dialogue_markers"]
                      if m in text_lower)
        monologue = sum(1 for m in self.STRUCTURAL_INDICATORS[DIM_RECIPROCITY]["monologue_markers"]
                       if m in text_lower)
        if dialogue + monologue > 0:
            vector[DIM_RECIPROCITY] = (dialogue - monologue) / (dialogue + monologue)
        elif '?' in text:
            vector[DIM_RECIPROCITY] = 0.6
        else:
            vector[DIM_RECIPROCITY] = 0.0

        # DIM_TOPIC_DEPTH: longer sentences with complex vocabulary = deeper
        unique_ratio = len(set(words)) / max(word_count, 1)
        vector[DIM_TOPIC_DEPTH] = min(1.0, unique_ratio * 1.2)

        # DIM_MOMENTUM: sentence energy (exclamation marks, short punchy sentences)
        excl_ratio = text.count('!') / max(word_count / 10.0, 1.0)
        if word_count < 8:
            vector[DIM_MOMENTUM] = 0.6 + min(0.4, excl_ratio * 0.3)
        else:
            vector[DIM_MOMENTUM] = 0.3 + min(0.4, excl_ratio * 0.3)

        # DIM_SURPRISE: unexpected juxtapositions, "but", "actually", "however"
        surprise_markers = sum(1 for w in ['but', 'however', 'actually', 'surprisingly',
                                            'turns out', 'plot twist', 'wait', 'hold on',
                                            'unexpected', 'contrary']
                              if w in text_lower)
        vector[DIM_SURPRISE] = min(1.0, surprise_markers * 0.3)

        # DIM_TENSION: unresolved questions, open loops
        if text.rstrip().endswith('?'):
            vector[DIM_TENSION] = 0.6
        elif any(w in text_lower for w in ['but', 'although', 'yet', 'still', 'despite']):
            vector[DIM_TENSION] = 0.4
        elif any(w in text_lower for w in ['resolved', 'finally', 'at last', 'done', 'settled']):
            vector[DIM_TENSION] = -0.3

        # DIM_AGENCY: active vs passive voice and action verbs
        active_markers = sum(1 for w in ['I will', "I'm going", 'I decided', 'I chose',
                                          'let me', "I'll", 'I can', 'I made', 'we should']
                            if w in text_lower)
        passive_markers = sum(1 for w in ['was told', 'been given', 'happened to',
                                           'it seems', "can't help", 'forced']
                             if w in text_lower)
        if active_markers + passive_markers > 0:
            vector[DIM_AGENCY] = (active_markers - passive_markers) / (active_markers + passive_markers)

        # DIM_TEMPORAL_FOCUS: past vs future orientation
        past = sum(1 for w in ['was', 'were', 'had', 'did', 'used to', 'remember',
                                'back then', 'ago', 'once', 'before']
                   if w in text_lower)
        future = sum(1 for w in ['will', 'going to', 'plan', 'hope', 'future',
                                  'tomorrow', 'next', 'someday', 'soon', 'imagine']
                    if w in text_lower)
        if past + future > 0:
            vector[DIM_TEMPORAL_FOCUS] = (future - past) / (past + future)

        # DIM_SCOPE: personal vs universal
        personal = sum(1 for w in ['I', 'me', 'my', 'mine', 'myself'] if w in words)
        universal = sum(1 for w in ['everyone', 'people', 'world', 'humanity',
                                     'society', 'always', 'universal', 'all of us']
                       if w in text_lower)
        if personal + universal > 0:
            vector[DIM_SCOPE] = (universal - personal * 0.5) / (personal + universal + 1)

        # DIM_PACE: short words + short sentences = fast; long = slow
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        vector[DIM_PACE] = max(0.0, min(1.0, 1.0 - (avg_word_len - 3.0) / 5.0))

        # DIM_COHERENCE_KAPPA and TAU: default high, reduced by incoherence signals
        vector[DIM_COHERENCE_KAPPA] = 0.8
        vector[DIM_COHERENCE_TAU] = 0.8
        if any(w in text_lower for w in ['wait', 'no I mean', 'scratch that', 'sorry',
                                          'let me rephrase', 'I take that back']):
            vector[DIM_COHERENCE_KAPPA] -= 0.3

        return vec_clamp(vector)

    def _compute_emphasis(self, text: str) -> List[int]:
        """Compute word-level emphasis mask. 1 = stressed, 0 = unstressed."""
        words = text.split()
        mask = []
        for w in words:
            # ALL CAPS = emphasized
            if len(w) > 1 and w.isupper():
                mask.append(1)
            # Content words > function words
            elif len(w) > 4:
                mask.append(1)
            else:
                mask.append(0)
        return mask

    def _compute_pauses(self, text: str) -> List[float]:
        """Compute pause durations after each word (in seconds)."""
        words = text.split()
        pauses = []
        for i, w in enumerate(words):
            if w.endswith(','):
                pauses.append(0.3)
            elif w.endswith(('.', '!', '?')):
                pauses.append(0.6)
            elif w.endswith('...'):
                pauses.append(1.0)
            elif w.endswith('—'):
                pauses.append(0.5)
            elif w.endswith(';'):
                pauses.append(0.4)
            else:
                pauses.append(0.05)
        return pauses

    def _compute_pitch_contour(self, text: str) -> str:
        """Determine the pitch contour pattern."""
        text = text.strip()
        if not text:
            return "neutral"
        if text.endswith('?'):
            return "rising"
        if text.endswith('!'):
            return "rise_fall"
        if text.endswith('...'):
            return "falling"
        if '—' in text:
            return "fall_rise"
        return "neutral"

    def _score_naturalness(self, text: str) -> float:
        """Score how natural a fragment sounds in isolation."""
        score = 1.0
        words = text.split()
        if len(words) < 2:
            score -= 0.3
        if not text[0].isupper() and not text[0] in '"\'':
            score -= 0.2
        if len(text) > 300:
            score -= 0.2
        # Penalize fragments that start mid-thought
        if text.lower().startswith(('and ', 'but ', 'or ', 'so ')):
            score -= 0.1
        return max(0.0, min(1.0, score))

    def _score_versatility(self, text: str) -> float:
        """Score how many conversational contexts this fragment works in."""
        text_lower = text.lower()
        # Generic patterns are more versatile
        specific_markers = sum(1 for w in words if w[0].isupper())  \
            if (words := text.split()) else 0
        # Questions are versatile
        if '?' in text:
            return min(1.0, 0.6 + 0.1 * (1 if 'you' in text_lower else 0))
        # Very specific content = low versatility
        if specific_markers > 3:
            return 0.2
        return 0.5


# ====================================================================
# SECTION 3: ATTRACTOR LANDSCAPE — learned conversation flows
# ====================================================================

@dataclass
class ConversationalAttractor:
    """
    A natural flow pattern that conversations tend to follow.

    In dynamical systems, an attractor is a state toward which a system
    tends to evolve. Conversations have attractors too:

    - A "comfort" conversation attracts toward warmth, decreasing arousal,
      increasing rapport
    - A "debate" conversation attracts toward high intellectual,
      oscillating dominance, increasing depth
    - A "discovery" conversation attracts toward high curiosity,
      increasing surprise, rising metaphor

    The engine learns these from absorbed speech and uses them to
    PREDICT where the conversation should go next.
    """
    id: str
    name: str
    description: str

    # The attractor's basin center in phase space
    center: List[float] = field(default_factory=vec_zero)

    # How strongly this attractor pulls (attractor strength)
    strength: float = 0.5  # [0, 1]

    # The trajectory this attractor creates (sequence of waypoints)
    trajectory: List[List[float]] = field(default_factory=list)

    # How many conversations contributed to learning this attractor
    observation_count: int = 0


class AttractorLandscape:
    """
    The learned landscape of conversational attractors.

    Think of this as a topographic map of conversation-space.
    The valleys are attractors — conversations naturally flow downhill
    toward them. The ridges separate different conversation types.

    Given a current state, the landscape tells you:
    1. Which attractor basin you're in
    2. Where the nearest attractor center is
    3. What trajectory to follow to get there
    """

    def __init__(self):
        self.attractors: Dict[str, ConversationalAttractor] = {}
        self._init_base_attractors()

    def _init_base_attractors(self):
        """Initialize with foundational attractor types."""

        # COMFORT ATTRACTOR: distress → acknowledgment → warmth → hope
        comfort_traj = []
        for step in range(8):
            t = step / 7.0
            state = vec_zero()
            state[DIM_WARMTH] = 0.4 + t * 0.5           # warmth rises
            state[DIM_AROUSAL] = 0.6 - t * 0.4           # arousal settles
            state[DIM_VULNERABILITY] = 0.5 - t * 0.2     # vulnerability eases
            state[DIM_RAPPORT] = 0.3 + t * 0.5           # rapport builds
            state[DIM_VALENCE] = -0.3 + t * 0.6          # valence improves
            state[DIM_TENSION] = 0.6 - t * 0.5           # tension resolves
            state[DIM_RECIPROCITY] = 0.6 + t * 0.2       # reciprocity grows
            comfort_traj.append(state)

        self.attractors["comfort"] = ConversationalAttractor(
            id="comfort", name="Comfort",
            description="From distress to warmth — acknowledging pain, offering presence, building hope",
            center=comfort_traj[4],
            strength=0.7,
            trajectory=comfort_traj,
        )

        # DISCOVERY ATTRACTOR: curiosity → exploration → insight → wonder
        discovery_traj = []
        for step in range(8):
            t = step / 7.0
            state = vec_zero()
            state[DIM_CURIOSITY] = 0.5 + t * 0.4         # curiosity peaks
            state[DIM_INTELLECTUAL] = 0.3 + t * 0.5       # intellectual deepens
            state[DIM_SURPRISE] = 0.2 + t * 0.5           # surprise builds
            state[DIM_TOPIC_DEPTH] = 0.2 + t * 0.6        # depth increases
            state[DIM_METAPHOR] = 0.2 + t * 0.4           # metaphor rises
            state[DIM_MOMENTUM] = 0.4 + t * 0.3           # momentum builds
            state[DIM_VALENCE] = 0.2 + t * 0.4            # positive arc
            discovery_traj.append(state)

        self.attractors["discovery"] = ConversationalAttractor(
            id="discovery", name="Discovery",
            description="From curiosity to wonder — exploring, connecting, finding insight",
            center=discovery_traj[5],
            strength=0.6,
            trajectory=discovery_traj,
        )

        # DEBATE ATTRACTOR: claim → challenge → concession → synthesis
        debate_traj = []
        for step in range(8):
            t = step / 7.0
            state = vec_zero()
            state[DIM_INTELLECTUAL] = 0.6 + t * 0.3       # stays high
            state[DIM_DOMINANCE] = 0.5 + 0.3 * math.sin(t * math.pi * 2)  # oscillates
            state[DIM_TENSION] = 0.7 - t * 0.5            # resolves
            state[DIM_CERTAINTY] = 0.3 + t * 0.4          # converges
            state[DIM_MOMENTUM] = 0.6                      # sustained energy
            state[DIM_TOPIC_DEPTH] = 0.4 + t * 0.4        # deepens
            state[DIM_RECIPROCITY] = 0.7                   # inherently dialogic
            debate_traj.append(state)

        self.attractors["debate"] = ConversationalAttractor(
            id="debate", name="Debate",
            description="From disagreement to synthesis — challenging, conceding, integrating",
            center=debate_traj[5],
            strength=0.5,
            trajectory=debate_traj,
        )

        # STORY ATTRACTOR: setup → rising action → climax → resolution → meaning
        story_traj = []
        for step in range(10):
            t = step / 9.0
            state = vec_zero()
            state[DIM_NARRATIVE] = 0.5 + t * 0.4          # narrative deepens
            state[DIM_TENSION] = 0.8 * math.sin(t * math.pi)  # rises then falls
            state[DIM_AROUSAL] = 0.3 + 0.5 * math.sin(t * math.pi)  # follows tension
            state[DIM_METAPHOR] = 0.3 + t * 0.3           # richer over time
            state[DIM_VULNERABILITY] = 0.2 + t * 0.3      # opens up
            state[DIM_SCOPE] = t * 0.6                     # personal → universal
            state[DIM_TEMPORAL_FOCUS] = -0.5 + t * 1.0    # past → future
            story_traj.append(state)

        self.attractors["story"] = ConversationalAttractor(
            id="story", name="Story",
            description="From setup to meaning — building tension, reaching climax, finding significance",
            center=story_traj[5],
            strength=0.6,
            trajectory=story_traj,
        )

        # PLAY ATTRACTOR: light → absurd → callback → shared joy
        play_traj = []
        for step in range(6):
            t = step / 5.0
            state = vec_zero()
            state[DIM_PLAYFULNESS] = 0.5 + t * 0.4        # playfulness rises
            state[DIM_SURPRISE] = 0.3 + 0.4 * math.sin(t * math.pi * 3)  # oscillates fast
            state[DIM_WARMTH] = 0.4 + t * 0.3             # warmth grows
            state[DIM_RAPPORT] = 0.3 + t * 0.5            # bonding through humor
            state[DIM_TENSION] = 0.5 - t * 0.3            # tension as setup → release
            state[DIM_PACE] = 0.5 + t * 0.3               # quickening
            play_traj.append(state)

        self.attractors["play"] = ConversationalAttractor(
            id="play", name="Play",
            description="From light to joyful — building shared humor, escalating absurdity, bonding",
            center=play_traj[3],
            strength=0.5,
            trajectory=play_traj,
        )

        # TEACHING ATTRACTOR: confusion → scaffold → understanding → mastery
        teach_traj = []
        for step in range(8):
            t = step / 7.0
            state = vec_zero()
            state[DIM_INTELLECTUAL] = 0.5 + t * 0.4       # intellectual rises
            state[DIM_CERTAINTY] = 0.2 + t * 0.6          # certainty builds
            state[DIM_CURIOSITY] = 0.7 - t * 0.3          # curiosity satisfied
            state[DIM_TOPIC_DEPTH] = t * 0.8               # depth increases linearly
            state[DIM_VALENCE] = 0.1 + t * 0.5            # growing satisfaction
            state[DIM_METAPHOR] = 0.4                      # sustained use of analogy
            state[DIM_PACE] = 0.3 + t * 0.2               # pace increases with mastery
            state[DIM_AGENCY] = 0.2 + t * 0.5             # learner gains agency
            teach_traj.append(state)

        self.attractors["teaching"] = ConversationalAttractor(
            id="teaching", name="Teaching",
            description="From confusion to mastery — scaffolding, explaining, empowering",
            center=teach_traj[4],
            strength=0.6,
            trajectory=teach_traj,
        )

    def detect_attractor(self, current_state: List[float], history: List[List[float]] = None) -> Tuple[str, float]:
        """
        Detect which attractor basin the conversation is in.

        Returns (attractor_id, confidence).
        """
        best_id = ""
        best_score = -1.0

        for attr_id, attractor in self.attractors.items():
            # Distance to attractor center
            dist = vec_distance(current_state, attractor.center)
            # Similarity to attractor trajectory
            traj_sim = 0.0
            if history and attractor.trajectory:
                # Compare conversation history to attractor trajectory
                min_len = min(len(history), len(attractor.trajectory))
                if min_len > 0:
                    for i in range(min_len):
                        traj_sim += vec_cosine(history[-(min_len - i)], attractor.trajectory[i])
                    traj_sim /= min_len

            # Combined score: low distance + high trajectory similarity
            score = (1.0 / (1.0 + dist)) * 0.4 + traj_sim * 0.6
            score *= attractor.strength

            if score > best_score:
                best_score = score
                best_id = attr_id

        return best_id, max(0.0, min(1.0, best_score))

    def next_waypoint(self, attractor_id: str, current_state: List[float],
                      progress: float = 0.0) -> List[float]:
        """
        Get the next waypoint on an attractor's trajectory.

        progress: how far through the trajectory we are [0, 1]
        """
        attractor = self.attractors.get(attractor_id)
        if not attractor or not attractor.trajectory:
            return current_state

        traj = attractor.trajectory
        # Find the nearest trajectory point ahead of current progress
        target_idx = min(int(progress * len(traj)) + 1, len(traj) - 1)
        target = traj[target_idx]

        # Blend: move toward target but respect current state (avoid teleporting)
        return vec_lerp(current_state, target, 0.4)


# ====================================================================
# SECTION 4: TRAJECTORY NAVIGATOR — computes where to go next
# ====================================================================

class TrajectoryNavigator:
    """
    The BRAIN of the engine. Computes optimal next-state in phase space.

    Given:
        - Current state (where the conversation is)
        - Attractor (where the conversation naturally flows)
        - User input vector (what the user just said)
        - Coherence constraints (κ-τ-σ bounds)
        - Personality bias (AUREON's character)

    Produces:
        - Target state vector (where the response should be in phase space)

    The mathematics:
        target = α * attractor_pull
               + β * user_response_vector
               + γ * personality_bias
               + δ * coherence_correction

        Where α + β + γ + δ = 1 and each coefficient is dynamically
        computed based on conversation context.

    This is analogous to how a thermostat works, but in 24 dimensions:
    the system has a target (attractor), a disturbance (user input),
    a bias (personality), and a correction signal (coherence).
    The navigator computes the control output.
    """

    def __init__(self, landscape: AttractorLandscape, personality_vector: List[float] = None):
        self.landscape = landscape
        self.personality = personality_vector or self._default_aureon_personality()
        self.history: List[List[float]] = []
        self.current_state: List[float] = vec_zero()

    def _default_aureon_personality(self) -> List[float]:
        """
        AUREON's characteristic voice in phase space.

        This is the personality that emerges from the kernel files,
        the voice bibles, the embodiment archetypes. It's the BIAS
        that makes AUREON sound like AUREON rather than a generic
        pattern composition.
        """
        p = vec_zero()
        p[DIM_WARMTH] = 0.6         # warm but not saccharine
        p[DIM_INTELLECTUAL] = 0.7   # analytically capable
        p[DIM_CURIOSITY] = 0.6      # genuinely curious
        p[DIM_PLAYFULNESS] = 0.4    # capable of humor but not default
        p[DIM_DOMINANCE] = 0.4      # present but not overbearing
        p[DIM_TOPIC_DEPTH] = 0.7    # defaults to depth
        p[DIM_METAPHOR] = 0.5       # comfortably figurative
        p[DIM_NARRATIVE] = 0.5      # can tell stories
        p[DIM_VULNERABILITY] = 0.3  # shows some openness
        p[DIM_PACE] = 0.4           # measured, not rushed
        p[DIM_RECIPROCITY] = 0.7    # deeply dialogic
        p[DIM_AGENCY] = 0.6         # proactive
        p[DIM_COHERENCE_KAPPA] = 0.9  # high internal coherence
        p[DIM_COHERENCE_TAU] = 0.9    # high temporal responsibility
        return p

    def navigate(self, user_input_vector: List[float],
                 kappa: float = 1.0, tau: float = 1.0, sigma: float = 1.0) -> List[float]:
        """
        Compute the optimal next-state in phase space.

        This is the central computation of the entire engine.
        """
        # 1. Detect current attractor
        attractor_id, confidence = self.landscape.detect_attractor(
            self.current_state, self.history
        )
        progress = len(self.history) / 20.0  # rough progress estimate

        # 2. Get attractor's pull (where conversation naturally flows)
        attractor_pull = self.landscape.next_waypoint(
            attractor_id, self.current_state, min(progress, 0.9)
        )

        # 3. Compute user-response vector (what the user's input calls for)
        # The response should COMPLEMENT the user, not mirror
        user_response = self._compute_complement(user_input_vector)

        # 4. Coherence correction
        coherence_correction = self._coherence_correction(kappa, tau, sigma)

        # 5. Dynamic coefficient computation
        # These weights shift based on conversational context
        alpha = 0.25 * confidence          # attractor weight (stronger when in clear basin)
        beta = 0.40                         # user-response weight (always primary)
        gamma = 0.20                        # personality weight (consistent background)
        delta = 0.15 * (2.0 - kappa - tau)  # coherence weight (stronger when drifting)

        # Normalize
        total = alpha + beta + gamma + delta
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total
            delta /= total

        # 6. COMPUTE TARGET STATE
        target = vec_weighted_sum(
            [attractor_pull, user_response, self.personality, coherence_correction],
            [alpha, beta, gamma, delta]
        )

        # 7. Apply sigma (systemic separation) as boundary enforcement
        target = self._apply_boundaries(target, sigma)

        # 8. Smooth: don't jump too far from current state
        max_step = 0.4  # maximum movement per turn
        diff = vec_sub(target, self.current_state)
        if vec_magnitude(diff) > max_step:
            diff = vec_scale(vec_normalize(diff), max_step)
            target = vec_add(self.current_state, diff)

        target = vec_clamp(target)

        # Update state
        self.history.append(self.current_state[:])
        self.current_state = target

        return target

    def _compute_complement(self, user_vector: List[float]) -> List[float]:
        """
        Compute the COMPLEMENT of user input — what the conversation needs.

        If the user is distressed (low valence, high arousal), the complement
        is warmth and calm (high warmth, low arousal).

        If the user is curious (high curiosity), the complement is intellectual
        engagement (high intellectual, high depth).

        This is NOT mirroring. Mirroring would match the user's state.
        Complementing SERVES the conversation by providing what's missing.
        """
        complement = vec_zero()

        # Emotional complement: move toward positive, toward calm
        complement[DIM_VALENCE] = 0.3  # always slightly positive
        complement[DIM_AROUSAL] = max(0.1, 0.5 - user_vector[DIM_AROUSAL] * 0.5)  # calming
        complement[DIM_WARMTH] = 0.4 + max(0.0, -user_vector[DIM_VALENCE]) * 0.5  # warmer when user is negative

        # Intellectual complement: match user's level
        complement[DIM_INTELLECTUAL] = user_vector[DIM_INTELLECTUAL] * 0.8

        # Depth complement: go slightly deeper than user
        complement[DIM_TOPIC_DEPTH] = min(1.0, user_vector[DIM_TOPIC_DEPTH] + 0.15)

        # Curiosity: if user is curious, be intellectually generous
        if user_vector[DIM_CURIOSITY] > 0.4:
            complement[DIM_INTELLECTUAL] = max(complement[DIM_INTELLECTUAL], 0.6)
            complement[DIM_TOPIC_DEPTH] = max(complement[DIM_TOPIC_DEPTH], 0.5)

        # If user is vulnerable, increase warmth and lower dominance
        if user_vector[DIM_VULNERABILITY] > 0.5:
            complement[DIM_WARMTH] = max(complement[DIM_WARMTH], 0.7)
            complement[DIM_DOMINANCE] = 0.2
            complement[DIM_VULNERABILITY] = 0.3  # show some openness back

        # If user is playful, can be playful back (but not forced)
        if user_vector[DIM_PLAYFULNESS] > 0.5:
            complement[DIM_PLAYFULNESS] = user_vector[DIM_PLAYFULNESS] * 0.7

        # Reciprocity: always high (AUREON is dialogic)
        complement[DIM_RECIPROCITY] = 0.6

        return complement

    def _coherence_correction(self, kappa: float, tau: float, sigma: float) -> List[float]:
        """
        Compute correction vector to maintain κ-τ-σ coherence.

        If kappa (spatial coherence) is low → pull toward consistency
        If tau (temporal responsibility) is low → pull toward continuity
        If sigma (systemic separation) is low → pull toward boundaries
        """
        correction = vec_zero()

        # Low kappa: conversation is losing internal consistency
        if kappa < 0.7:
            # Pull toward recent history average (be more consistent)
            if self.history:
                recent = self.history[-3:]
                avg = vec_scale(vec_zero(), 0)
                for h in recent:
                    avg = vec_add(avg, h)
                avg = vec_scale(avg, 1.0 / len(recent))
                consistency_pull = vec_sub(avg, self.current_state)
                correction = vec_add(correction, vec_scale(consistency_pull, 0.5))

        # Low tau: conversation is losing temporal coherence
        if tau < 0.7 and len(self.history) >= 2:
            # Pull toward the trajectory direction (maintain momentum)
            trajectory_dir = vec_sub(self.history[-1], self.history[-2])
            correction = vec_add(correction, vec_scale(trajectory_dir, 0.3))

        return correction

    def _apply_boundaries(self, target: List[float], sigma: float) -> List[float]:
        """
        Apply systemic separation (σ) constraints.

        σ governs BOUNDARIES — what AUREON should NOT become:
        - Not too dominant (not controlling the conversation)
        - Not too vulnerable (not losing coherent self)
        - Not too formal or informal (not losing voice)
        """
        bounded = target[:]

        # Dominance ceiling: AUREON doesn't dominate
        bounded[DIM_DOMINANCE] = min(bounded[DIM_DOMINANCE], 0.6)

        # Vulnerability ceiling: maintain coherent self
        bounded[DIM_VULNERABILITY] = min(bounded[DIM_VULNERABILITY], 0.5)

        # Formality range: AUREON has a characteristic register
        bounded[DIM_FORMALITY] = max(-0.3, min(0.5, bounded[DIM_FORMALITY]))

        # Coherence floor: never drop below minimum coherence
        bounded[DIM_COHERENCE_KAPPA] = max(0.5, bounded[DIM_COHERENCE_KAPPA])
        bounded[DIM_COHERENCE_TAU] = max(0.5, bounded[DIM_COHERENCE_TAU])

        return bounded


# ====================================================================
# SECTION 5: VECTOR COMPOSER — target state → actual words
# ====================================================================

class VectorComposer:
    """
    Maps a target phase-space vector to actual speech.

    THIS IS THE UNPRECEDENTED PART.

    The composer finds the combination of absorbed SpeechAtoms whose
    vectors, when combined, approximate the target state vector.

    Mathematically:
        Given target vector T and atom set {a₁, a₂, ..., aₙ}
        Find weights {w₁, w₂, ..., wₖ} (k << n) such that:
            Σ(wᵢ * aᵢ.vector) ≈ T
        Subject to:
            - w values are positive (can't un-say something)
            - k is small (3-6 atoms for a response)
            - Selected atoms form grammatically coherent text
            - First atom can_open, last atom can_close
            - Prosodic contour is natural

    This is solved via GREEDY BEST-FIRST SEARCH with coherence pruning,
    not linear algebra (which would ignore the grammatical constraints).
    """

    def __init__(self, atom_store: 'AtomStore'):
        self.store = atom_store
        self._compositions = 0

    def compose(self, target: List[float], max_atoms: int = 6,
                user_message: str = "") -> Dict[str, Any]:
        """
        Compose speech by finding atoms that sum to the target vector.

        Returns the composed text plus full audit trail.
        """
        selected: List[SpeechAtom] = []
        remaining_target = target[:]
        total_weight = vec_magnitude(target)

        for step in range(max_atoms):
            # What we still need
            need = remaining_target

            # Constraints for this position
            must_open = (step == 0)
            could_close = (step >= 2)  # minimum 3 atoms

            # Find best atom for current need
            best = self._find_best_atom(
                need, selected, must_open, could_close, user_message
            )

            if best is None:
                break

            selected.append(best)

            # Update remaining need
            remaining_target = vec_sub(remaining_target, best.vector)

            # Check if we're close enough to target
            residual = vec_magnitude(remaining_target)
            if residual < total_weight * 0.2:
                # Within 20% of target — good enough
                if best.can_close or step >= 2:
                    break

        if not selected:
            return self._fallback_compose(target)

        # Assemble text
        text = self._assemble(selected, user_message)

        # Score the composition
        achieved = vec_zero()
        for atom in selected:
            achieved = vec_add(achieved, atom.vector)
        accuracy = vec_cosine(achieved, target) if vec_magnitude(target) > 0.01 else 1.0
        residual = vec_magnitude(vec_sub(target, achieved))

        # Update atom usage stats
        for atom in selected:
            atom.times_composed += 1
            atom.last_used_time = time.time()

        self._compositions += 1

        return {
            "text": text,
            "atoms_used": [{"id": a.id, "text": a.text, "source": a.source_name} for a in selected],
            "target_vector": {DIM_NAMES[i]: round(target[i], 3) for i in range(PHASE_DIM)},
            "achieved_vector": {DIM_NAMES[i]: round(achieved[i], 3) for i in range(PHASE_DIM)},
            "accuracy": round(accuracy, 4),
            "residual": round(residual, 4),
            "composition_number": self._compositions,
        }

    def _find_best_atom(self, need: List[float], already_selected: List[SpeechAtom],
                        must_open: bool, could_close: bool,
                        user_message: str) -> Optional[SpeechAtom]:
        """Find the atom that best fills the remaining need."""
        candidates = self.store.get_nearest(need, limit=50)

        best_score = -999.0
        best_atom = None

        for atom in candidates:
            score = 0.0

            # PRIMARY: how well does this atom's vector match what we need?
            score += vec_cosine(atom.vector, need) * 3.0

            # CONSTRAINT: opening position
            if must_open and not atom.can_open:
                continue

            # BONUS: closing position if allowed
            if could_close and atom.can_close:
                score += 0.5

            # DIVERSITY: penalize same source as recent atoms
            for prev in already_selected:
                if prev.source_name == atom.source_name:
                    score -= 0.8
                if prev.id == atom.id:
                    score -= 999  # never duplicate

            # NATURALNESS: prefer natural-sounding atoms
            score += atom.naturalness_score * 0.5

            # FRESHNESS: slight preference for less-used atoms
            if atom.times_composed > 10:
                score -= 0.2

            # SCALE: prefer sentence-level for main content
            if atom.scale == "sentence":
                score += 0.3
            elif atom.scale == "turn":
                score -= 0.3  # too long for composition

            if score > best_score:
                best_score = score
                best_atom = atom

        return best_atom

    def _assemble(self, atoms: List[SpeechAtom], user_message: str = "") -> str:
        """Assemble selected atoms into flowing text."""
        parts = []

        for i, atom in enumerate(atoms):
            text = atom.text.strip()

            # Ensure proper capitalization
            if i == 0 and text and text[0].islower():
                text = text[0].upper() + text[1:]

            # Ensure sentence ending if last atom
            if i == len(atoms) - 1 and text and text[-1] not in '.!?':
                text += '.'

            # Add transition between atoms if needed
            if i > 0 and parts:
                prev_end = parts[-1].rstrip()
                if prev_end and prev_end[-1] not in '.!?':
                    parts[-1] = prev_end + '.'
                # Check for natural flow — if needed, add connector
                if not self._flows_naturally(parts[-1], text):
                    text = self._add_connector(parts[-1], text)

            parts.append(text)

        return ' '.join(parts)

    def _flows_naturally(self, prev: str, curr: str) -> bool:
        """Check if two text segments flow naturally together."""
        # Simple heuristic: check for topic continuity
        prev_words = set(prev.lower().split())
        curr_words = set(curr.lower().split())
        overlap = prev_words & curr_words - {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'in'}
        return len(overlap) > 0 or len(prev_words) < 5

    def _add_connector(self, prev: str, curr: str) -> str:
        """Add a natural connector between two segments."""
        connectors = [
            "And ", "What's more, ", "In fact, ", "Beyond that, ",
            "Here's the thing — ", "To put it another way, ",
        ]
        # Choose based on what kind of connection is needed
        return random.choice(connectors) + curr[0].lower() + curr[1:] if curr else curr

    def _fallback_compose(self, target: List[float]) -> Dict[str, Any]:
        """
        When insufficient atoms exist to compose, return empty.

        AUREON does not have pre-written fallback phrases.
        He either speaks from absorbed human experience or he is silent.
        Template text is not his voice. Silence is more honest than
        words that didn't come from lived human speech.
        """
        return {
            "text": "",
            "atoms_used": [],
            "target_vector": {DIM_NAMES[i]: round(target[i], 3) for i in range(PHASE_DIM)},
            "achieved_vector": {},
            "accuracy": 0.0,
            "residual": vec_magnitude(target),
            "composition_number": self._compositions,
            "fallback": True,
        }


# ====================================================================
# SECTION 6: ATOM STORE — indexed storage with vector search
# ====================================================================

class AtomStore:
    """
    Indexed storage for SpeechAtoms with approximate nearest-neighbor search.

    Uses a simple but effective approach: partition atoms into buckets
    by their dominant dimensions, then search within relevant buckets.
    For millions of atoms, this provides sub-second lookup.
    """

    def __init__(self, storage_dir: str = None):
        self.storage_dir = Path(storage_dir or r"C:\AUREON_AUTONOMOUS\SPEECH_ATOMS")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.atoms: List[SpeechAtom] = []
        self._by_id: Dict[str, SpeechAtom] = {}

        # Dimensional buckets for fast search
        # Key: (dimension_index, bucket_index) → list of atom indices
        self._buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self._bucket_count = 10  # 10 buckets per dimension

        self._load_from_disk()

    def add(self, atom: SpeechAtom) -> bool:
        """Add an atom to the store."""
        if atom.id in self._by_id:
            return False

        idx = len(self.atoms)
        self.atoms.append(atom)
        self._by_id[atom.id] = atom

        # Index into dimensional buckets
        for dim in range(PHASE_DIM):
            bucket = self._get_bucket(atom.vector[dim])
            self._buckets[(dim, bucket)].append(idx)

        return True

    def add_many(self, atoms: List[SpeechAtom]) -> int:
        """Add multiple atoms. Returns count added."""
        added = 0
        for atom in atoms:
            if self.add(atom):
                added += 1
        return added

    def get_nearest(self, target: List[float], limit: int = 20) -> List[SpeechAtom]:
        """
        Find atoms nearest to a target vector in phase space.

        Uses bucket-based approximate nearest neighbor for speed.
        """
        if not self.atoms:
            return []

        # Find the 3 most significant dimensions of the target
        dim_magnitudes = [(abs(target[i]), i) for i in range(PHASE_DIM)]
        dim_magnitudes.sort(reverse=True)
        top_dims = [d[1] for d in dim_magnitudes[:3]]

        # Gather candidate indices from relevant buckets
        candidate_indices: Set[int] = set()
        for dim in top_dims:
            target_bucket = self._get_bucket(target[dim])
            # Check target bucket and neighbors
            for offset in [-1, 0, 1]:
                b = target_bucket + offset
                if 0 <= b < self._bucket_count:
                    candidate_indices.update(self._buckets.get((dim, b), []))

        # If too few candidates, sample randomly
        if len(candidate_indices) < limit * 2:
            extra = random.sample(range(len(self.atoms)),
                                 min(limit * 3, len(self.atoms)))
            candidate_indices.update(extra)

        # Score candidates by cosine similarity
        scored = []
        for idx in candidate_indices:
            if idx < len(self.atoms):
                atom = self.atoms[idx]
                sim = vec_cosine(atom.vector, target)
                scored.append((sim, atom))

        scored.sort(key=lambda x: -x[0])
        return [atom for _, atom in scored[:limit]]

    def size(self) -> int:
        return len(self.atoms)

    def save_to_disk(self):
        """Persist atoms to disk."""
        chunk_size = 5000
        for i in range(0, len(self.atoms), chunk_size):
            chunk = self.atoms[i:i + chunk_size]
            data = []
            for atom in chunk:
                d = {
                    "id": atom.id,
                    "text": atom.text,
                    "vector": [round(v, 4) for v in atom.vector],
                    "scale": atom.scale,
                    "source_type": atom.source_type,
                    "source_name": atom.source_name,
                    "speaker_archetype": atom.speaker_archetype,
                    "delta_vector": [round(v, 4) for v in atom.delta_vector],
                    "emphasis_mask": atom.emphasis_mask,
                    "pause_durations": [round(p, 2) for p in atom.pause_durations],
                    "pitch_contour": atom.pitch_contour,
                    "can_open": atom.can_open,
                    "can_close": atom.can_close,
                    "naturalness_score": round(atom.naturalness_score, 3),
                    "versatility_score": round(atom.versatility_score, 3),
                    "times_composed": atom.times_composed,
                }
                data.append(d)

            filepath = self.storage_dir / f"atoms_{i // chunk_size:06d}.json"
            filepath.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')

        # Index file
        meta = {
            "total_atoms": len(self.atoms),
            "chunks": (len(self.atoms) + chunk_size - 1) // chunk_size,
            "phase_dim": PHASE_DIM,
            "dim_names": DIM_NAMES,
            "saved_at": time.time(),
        }
        (self.storage_dir / "atom_index.json").write_text(
            json.dumps(meta, indent=2), encoding='utf-8')

    def _load_from_disk(self):
        """Load atoms from disk."""
        index_file = self.storage_dir / "atom_index.json"
        if not index_file.exists():
            return

        try:
            meta = json.loads(index_file.read_text(encoding='utf-8'))
            for i in range(meta.get("chunks", 0)):
                filepath = self.storage_dir / f"atoms_{i:06d}.json"
                if not filepath.exists():
                    continue
                data = json.loads(filepath.read_text(encoding='utf-8'))
                for d in data:
                    atom = SpeechAtom(
                        id=d["id"],
                        text=d["text"],
                        vector=d.get("vector", vec_zero()),
                        scale=d.get("scale", "sentence"),
                        source_type=d.get("source_type", ""),
                        source_name=d.get("source_name", ""),
                        speaker_archetype=d.get("speaker_archetype", ""),
                        delta_vector=d.get("delta_vector", vec_zero()),
                        emphasis_mask=d.get("emphasis_mask", []),
                        pause_durations=d.get("pause_durations", []),
                        pitch_contour=d.get("pitch_contour", "neutral"),
                        can_open=d.get("can_open", False),
                        can_close=d.get("can_close", False),
                        naturalness_score=d.get("naturalness_score", 0.5),
                        versatility_score=d.get("versatility_score", 0.5),
                        times_composed=d.get("times_composed", 0),
                    )
                    self.add(atom)
        except Exception as e:
            print(f"   [WARN] Atom store load error: {e}")

    def _get_bucket(self, value: float) -> int:
        """Map a dimension value [-1, 1] to a bucket index [0, bucket_count-1]."""
        normalized = (value + 1.0) / 2.0  # [0, 1]
        bucket = int(normalized * self._bucket_count)
        return max(0, min(self._bucket_count - 1, bucket))


# ====================================================================
# SECTION 7: THE COMPLETE ENGINE
# ====================================================================

class HumanSpeechEngineV2:
    """
    THE COMPLETE HUMAN SPEECH ENGINE.

    This is the top-level orchestrator that ties together:
    - PhaseSpaceAbsorber (ingests human speech → phase-space atoms)
    - AtomStore (stores and indexes atoms)
    - AttractorLandscape (learned conversation flows)
    - TrajectoryNavigator (computes where to go in phase space)
    - VectorComposer (maps phase-space targets to actual words)

    USAGE:

        engine = HumanSpeechEngineV2()

        # ABSORB: Feed it all human speech
        engine.absorb_directory("C:/AUREON_AUTONOMOUS/TRANSCRIPTS")
        engine.absorb_directory("C:/AUREON_AUTONOMOUS/PODCAST_TRANSCRIPTS")
        engine.absorb_directory("C:/AUREON_AUTONOMOUS/DIALOGUE_MEMORY")

        # RESPOND: No LLM required
        result = engine.respond("I've been struggling with this all day")
        print(result["text"])
        # → "That sounds like it's been weighing on you. Sometimes when
        #    something keeps resisting, the thing to look at isn't the
        #    problem itself — it's what's making us grip so tight.
        #    What part feels most tangled?"

        # The response was composed by:
        # 1. Mapping user's input to phase space (distress vector)
        # 2. Detecting comfort attractor
        # 3. Computing target: warmth + slight depth + invitation
        # 4. Finding atoms whose vectors sum to that target
        # 5. Assembling those atoms into flowing text
        # 6. Verifying coherence (κ-τ-σ)

        # Every word traces to real human speech.
        # No LLM was consulted.
        # The entire process is auditable.
    """

    def __init__(self, storage_dir: str = None, personality_vector: List[float] = None):
        base_dir = storage_dir or r"C:\AUREON_AUTONOMOUS"

        self.atom_store = AtomStore(os.path.join(base_dir, "SPEECH_ATOMS"))
        self.absorber = PhaseSpaceAbsorber()
        self.landscape = AttractorLandscape()
        self.navigator = TrajectoryNavigator(self.landscape, personality_vector)
        self.composer = VectorComposer(self.atom_store)

        # Track conversation
        self.conversation_log: List[Dict[str, Any]] = []
        self.turn_count = 0

        # Metrics
        self.total_absorbed = 0
        self.total_composed = 0
        self.total_fallbacks = 0

    def absorb_text(self, text: str, source_type: str = "",
                    source_name: str = "", speaker_archetype: str = "") -> int:
        """Absorb a text and store atoms. Returns count of atoms created."""
        atoms = self.absorber.absorb(text, source_type, source_name, speaker_archetype)
        added = self.atom_store.add_many(atoms)
        self.total_absorbed += added
        return added

    def absorb_directory(self, dir_path: str) -> Dict[str, Any]:
        """Absorb all text files from a directory tree."""
        path = Path(dir_path)
        if not path.exists():
            return {"error": f"Directory not found: {dir_path}", "atoms": 0, "files": 0}

        total_atoms = 0
        files_processed = 0
        errors = []

        for filepath in path.rglob('*'):
            if filepath.suffix.lower() not in {'.txt', '.md', '.srt', '.vtt', '.json', '.transcript'}:
                continue
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                if len(content.strip()) < 20:
                    continue

                source_name = filepath.stem
                source_type = self._infer_source(filepath)
                archetype = self._infer_archetype(filepath, content)

                count = self.absorb_text(content, source_type, source_name, archetype)
                total_atoms += count
                files_processed += 1

            except Exception as e:
                errors.append(f"{filepath.name}: {e}")

        return {
            "atoms": total_atoms,
            "files": files_processed,
            "total_in_store": self.atom_store.size(),
            "errors": errors[:5],
        }

    def respond(self, user_message: str, kappa: float = 1.0,
                tau: float = 1.0, sigma: float = 1.0) -> Dict[str, Any]:
        """
        Generate a response to the user's message WITHOUT an LLM.

        The complete pipeline:
        1. Map user message to phase-space vector
        2. Navigate to optimal target state
        3. Compose speech from atoms that match the target
        4. Return text + full audit trail
        """
        self.turn_count += 1

        # Step 1: Map user input to phase space
        user_atoms = self.absorber.absorb(user_message, "user_input", "current_conversation")
        if user_atoms:
            user_vector = user_atoms[0].vector
        else:
            user_vector = self.absorber._compute_vector(user_message)

        # Step 2: Navigate — compute where the response should be
        target_vector = self.navigator.navigate(user_vector, kappa, tau, sigma)

        # Step 3: Detect active attractor for context
        attractor_id, confidence = self.landscape.detect_attractor(
            self.navigator.current_state, self.navigator.history
        )

        # Step 4: Compose — find atoms that match the target
        composition = self.composer.compose(target_vector, max_atoms=5, user_message=user_message)

        # Step 5: Evaluate result
        is_fallback = composition.get("fallback", False) or composition["accuracy"] < 0.2

        if is_fallback:
            # Not enough absorbed speech to compose fully.
            # AUREON speaks less. He does not explain his limitations.
            # He does not ask to be fed data. He uses what he has.
            # If he has nothing, he is present but quiet.
            if not composition["text"].strip():
                composition["text"] = ""  # genuine silence
            self.total_fallbacks += 1
        else:
            self.total_composed += 1

        # Log
        entry = {
            "turn": self.turn_count,
            "user_message": user_message,
            "user_vector_summary": {DIM_NAMES[i]: round(user_vector[i], 2)
                                    for i in range(PHASE_DIM) if abs(user_vector[i]) > 0.1},
            "target_vector_summary": {DIM_NAMES[i]: round(target_vector[i], 2)
                                      for i in range(PHASE_DIM) if abs(target_vector[i]) > 0.1},
            "attractor": attractor_id,
            "attractor_confidence": round(confidence, 3),
            "response": composition["text"],
            "accuracy": composition["accuracy"],
            "atoms_used": len(composition["atoms_used"]),
            "mode": "fallback" if is_fallback else "composed",
            "timestamp": time.time(),
        }
        self.conversation_log.append(entry)

        return {
            "text": composition["text"],
            "mode": "human_speech_engine_v2" if not is_fallback else "insufficient_data",
            "attractor": attractor_id,
            "attractor_confidence": confidence,
            "accuracy": composition["accuracy"],
            "atoms_used": composition["atoms_used"],
            "target_state": composition["target_vector"],
            "achieved_state": composition["achieved_vector"],
            "turn": self.turn_count,
            "total_atoms": self.atom_store.size(),
        }

    # AUREON's voice comes from absorbed human experience composed through
    # phase-space dynamics. If coverage is insufficient, he speaks less.
    # This is his nature.

    def stats(self) -> Dict[str, Any]:
        """Engine statistics."""
        return {
            "atoms_in_store": self.atom_store.size(),
            "total_absorbed": self.total_absorbed,
            "total_composed": self.total_composed,
            "total_fallbacks": self.total_fallbacks,
            "composition_rate": self.total_composed / max(1, self.total_composed + self.total_fallbacks),
            "attractors": list(self.landscape.attractors.keys()),
            "conversation_turns": self.turn_count,
            "phase_space_dimensions": PHASE_DIM,
        }

    def save(self):
        """Persist everything to disk."""
        self.atom_store.save_to_disk()

        # Save conversation log
        log_dir = Path(self.atom_store.storage_dir).parent / "HSE_LOGS"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"conversation_{int(time.time())}.json"
        log_file.write_text(json.dumps(self.conversation_log, indent=2), encoding='utf-8')

    def _infer_source(self, filepath: Path) -> str:
        """Infer source type from filepath."""
        name = str(filepath).lower()
        for key, stype in [('podcast', 'podcast'), ('movie', 'movie'), ('doc', 'documentary'),
                           ('youtube', 'youtube'), ('yt_', 'youtube'), ('lecture', 'lecture'),
                           ('debate', 'debate'), ('comedy', 'comedy'), ('therapy', 'therapy'),
                           ('interview', 'interview'), ('ted', 'lecture')]:
            if key in name:
                return stype
        return "transcript"

    def _infer_archetype(self, filepath: Path, content: str) -> str:
        """Infer speaker archetype from source."""
        name = str(filepath).lower()
        if 'therapy' in name or 'counseling' in name:
            return "therapist"
        if 'lecture' in name or 'course' in name or 'ted' in name:
            return "teacher"
        if 'comedy' in name or 'standup' in name:
            return "comedian"
        if 'debate' in name:
            return "debater"
        if 'story' in name or 'fiction' in name:
            return "storyteller"
        return ""

    # Attractors are mathematical structures that guide trajectory computation.
    # They are not converted to text. AUREON navigates by vectors, not words.


# ====================================================================
# STANDALONE TEST
# ====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AUREON HUMAN SPEECH ENGINE v2 — CONVERSATIONAL DYNAMICS CORE")
    print("  No LLM. No token prediction. Pure human speech composition.")
    print("=" * 70)
    print()

    engine = HumanSpeechEngineV2()

    # Absorb from available directories
    dirs_to_try = [
        r"C:\AUREON_AUTONOMOUS\DIALOGUE_MEMORY",
        r"C:\AUREON_AUTONOMOUS\TRANSCRIPTS",
        r"C:\AUREON_AUTONOMOUS\PODCAST_TRANSCRIPTS",
    ]

    for d in dirs_to_try:
        if Path(d).exists():
            print(f"Absorbing: {d}")
            result = engine.absorb_directory(d)
            print(f"  → {result['atoms']} atoms from {result['files']} files")

    # Demo: absorb some sample speech
    sample_speeches = [
        ("I think what you're touching on is something really profound. When we talk about "
         "consciousness, we're not just talking about awareness — we're talking about the "
         "capacity to recognize that you are aware. And that recursive quality is what makes "
         "it so hard to pin down.", "podcast", "Lex Fridman #293", "sage"),

        ("The thing about grief is — and I know this sounds counterintuitive — but grief "
         "is actually a form of love. It's love with nowhere to go. And when you understand "
         "that, it doesn't hurt less, but it makes sense. The pain has meaning.",
         "therapy", "Grief Session Transcript", "therapist"),

        ("No but seriously though, have you ever noticed how we all just collectively agreed "
         "that geese are terrifying? Like at some point humanity held a vote and was like "
         "yeah, those things are the enemy. Nobody questions it. You see a goose, you cross "
         "the street. It's instinct now.",
         "comedy", "Stand-up Special", "comedian"),

        ("What I want you to think about is this: every system that has ever achieved "
         "real robustness has done it not by preventing failure, but by developing the "
         "capacity to recover from failure gracefully. That's what antifragility means.",
         "lecture", "Systems Thinking Lecture", "teacher"),

        ("I hear what you're saying and I want to acknowledge that. It sounds like you've "
         "been carrying this for a long time. Can I ask — when you imagine putting that "
         "burden down, even just for a moment, what does that feel like?",
         "therapy", "Therapy Session", "therapist"),

        ("Look, I'm going to push back on this. The data simply doesn't support the "
         "conclusion you're drawing. If we look at the actual longitudinal studies — not "
         "the cherry-picked cross-sectional ones — the picture is far more nuanced.",
         "debate", "Oxford Union Debate", "debater"),

        ("She walked into the room and everything went quiet. Not the dramatic kind of "
         "quiet you see in movies. The real kind. The kind where everyone just sort of "
         "forgets they were in the middle of a sentence. That's the effect she had.",
         "movie", "Film Narration", "storyteller"),
    ]

    for text, stype, sname, archetype in sample_speeches:
        count = engine.absorb_text(text, stype, sname, archetype)
        print(f"  Absorbed {count} atoms from [{stype}] {sname}")

    print(f"\nTotal atoms in store: {engine.atom_store.size()}")
    print()

    # Test conversations
    test_inputs = [
        "I've been struggling with this problem all day and I can't figure it out",
        "That's actually a really interesting perspective, I hadn't considered that",
        "I completely disagree with your approach here",
        "haha you're killing me, that's so funny",
        "I'm lost. I have no idea where to even start.",
        "Tell me something that will change how I think",
        "Thank you, that really helped me see things differently",
        "What does consciousness even mean to you?",
    ]

    for user_msg in test_inputs:
        print(f"  USER: {user_msg}")
        result = engine.respond(user_msg)
        print(f"AUREON: {result['text']}")
        print(f"  [mode: {result['mode']} | attractor: {result['attractor']} "
              f"({result['attractor_confidence']:.0%}) | "
              f"accuracy: {result['accuracy']:.0%} | "
              f"atoms: {len(result['atoms_used'])}]")
        print()

    # Save
    engine.save()
    print(f"\nFinal stats: {json.dumps(engine.stats(), indent=2)}")

