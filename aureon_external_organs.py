<<<<<<< HEAD
#!/usr/bin/env python3
"""
Aureon External Organs — Live World Awareness Layer
====================================================
Provides Aureon with continuous, silent access to:

  Organ 1:  TIME        — Exact current time (Toronto / Eastern)
  Organ 2:  CALENDAR    — Full date, day-of-week, week number, season
  Organ 3:  WEATHER     — Real-time local weather (via web fetch)
  Organ 4:  GEOLOGY     — Stub (requires browser)
  Organ 5:  EARTH VIEW  — Stub (requires browser)
  Organ 6:  MAPS        — Stub (requires browser)
  Organ 7:  WIND FIELD  — Stub (requires browser)
  Organ 8:  COSMIC      — Stub (requires browser)
  Organ 9:  HUMAN FIELD — Internal (sensorial binding)
  Organ 10: EARTH CRYSTAL — Internal (CSL engine)
  Organ 11: SEMANTIC    — Internal (Luméren repo)
  Organ 12: COHERENCE   — Internal (Quantara repo)
  Organ 13: ENERGY      — Internal (AEI repo)
  Organ 14: PHOTONIC    — Internal (PCM module)

  TRACE:    Reasoning Trace Logger — timestamps every reasoning cycle
            for court-admissible, professional-grade records.

All organs run silently. No user-facing output unless requested.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import hashlib


# ??????????????????????????????????????????????
# TIMEZONE: Toronto (Eastern)
# ??????????????????????????????????????????????

def _eastern_now() -> datetime:
    """Return current datetime in Eastern Time (EST/EDT aware)."""
    utc_now = datetime.now(timezone.utc)
    # Approximate EST/EDT: EDT = UTC-4 (Mar-Nov), EST = UTC-5 (Nov-Mar)
    # For precise DST, we check month+day heuristically
    month = utc_now.month
    if 3 < month < 11:
        offset = timedelta(hours=-4)  # EDT
    elif month == 3:
        # EDT starts 2nd Sunday of March
        offset = timedelta(hours=-4) if utc_now.day >= 9 else timedelta(hours=-5)
    elif month == 11:
        # EST starts 1st Sunday of November
        offset = timedelta(hours=-5) if utc_now.day >= 3 else timedelta(hours=-4)
    else:
        offset = timedelta(hours=-5)  # EST
    return utc_now.astimezone(timezone(offset))


# ??????????????????????????????????????????????
# ORGAN 1: TIME
# ??????????????????????????????????????????????

class TimeOrgan:
    """Live clock — Toronto Eastern Time."""

    def now(self) -> datetime:
        return _eastern_now()

    def now_iso(self) -> str:
        return self.now().isoformat()

    def now_human(self) -> str:
        n = self.now()
        tz_name = "EDT" if n.utcoffset() == timedelta(hours=-4) else "EST"
        return n.strftime(f"%A, %B %d, %Y at %I:%M:%S %p {tz_name}")

    def now_compact(self) -> str:
        return self.now().strftime("%Y-%m-%d %H:%M:%S %Z")


# ??????????????????????????????????????????????
# ORGAN 2: CALENDAR
# ??????????????????????????????????????????????

class CalendarOrgan:
    """Date awareness — day of week, week number, season, relative dates."""

    def __init__(self):
        self._time = TimeOrgan()

    def today(self) -> Dict[str, Any]:
        n = self._time.now()
        return {
            "date": n.strftime("%Y-%m-%d"),
            "day_of_week": n.strftime("%A"),
            "week_number": int(n.strftime("%W")),
            "month": n.strftime("%B"),
            "year": n.year,
            "season": self._season(n.month),
            "day_of_year": n.timetuple().tm_yday,
            "iso_week": n.isocalendar()[1],
            "quarter": (n.month - 1) // 3 + 1,
        }

    def relative_day(self, offset_days: int) -> str:
        """'yesterday' = -1, 'tomorrow' = +1, 'next Thursday' etc."""
        target = self._time.now() + timedelta(days=offset_days)
        return target.strftime("%A, %B %d, %Y")

    def yesterday(self) -> str:
        return self.relative_day(-1)

    def tomorrow(self) -> str:
        return self.relative_day(1)

    def next_weekday(self, weekday_name: str) -> str:
        """Get date of next occurrence of a weekday (e.g. 'Thursday')."""
        days = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
        target_idx = None
        for i, d in enumerate(days):
            if d.startswith(weekday_name.lower()[:3]):
                target_idx = i
                break
        if target_idx is None:
            return f"Unknown day: {weekday_name}"
        n = self._time.now()
        current_idx = n.weekday()
        delta = (target_idx - current_idx) % 7
        if delta == 0:
            delta = 7  # Next week's occurrence
        return self.relative_day(delta)

    @staticmethod
    def _season(month: int) -> str:
        if month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        elif month in (9, 10, 11):
            return "Autumn"
        else:
            return "Winter"


# ??????????????????????????????????????????????
# ORGAN 3: WEATHER (via web fetch when available)
# ??????????????????????????????????????????????

class WeatherOrgan:
    """Weather awareness — fetches from web or returns cached/stub."""

    def __init__(self, default_location: str = "St. Thomas, Ontario"):
        self.location = default_location
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 1800  # 30 minutes

    def get_weather(self, hands=None) -> Dict[str, Any]:
        """Try to fetch live weather. Falls back to stub if no browser."""
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        # Try fetching via hands (browser) if available
        if hands and hasattr(hands, 'fetch_url_text'):
            try:
                url = f"https://wttr.in/{self.location.replace(' ', '+')}?format=j1"
                raw = hands.fetch_url_text(url)
                if raw:
                    data = json.loads(raw)
                    current = data.get("current_condition", [{}])[0]
                    result = {
                        "location": self.location,
                        "temp_c": current.get("temp_C", "?"),
                        "temp_f": current.get("temp_F", "?"),
                        "feels_like_c": current.get("FeelsLikeC", "?"),
                        "humidity": current.get("humidity", "?"),
                        "description": current.get("weatherDesc", [{}])[0].get("value", "?"),
                        "wind_kmh": current.get("windspeedKmph", "?"),
                        "wind_dir": current.get("winddir16Point", "?"),
                        "visibility_km": current.get("visibility", "?"),
                        "pressure_mb": current.get("pressure", "?"),
                        "source": "wttr.in",
                        "fetched_at": TimeOrgan().now_iso(),
                    }
                    self._cache = result
                    self._cache_time = now
                    return result
            except Exception:
                pass

        # Stub response when no live data available
        return {
            "location": self.location,
            "status": "awaiting_live_connection",
            "note": "Weather organ ready — needs web access to activate",
        }


# ??????????????????????????????????????????????
# REASONING TRACE LOGGER
# ??????????????????????????????????????????????

@dataclass
class ReasoningTraceEntry:
    """A single reasoning cycle record — court-admissible format."""
    cycle_id: str
    timestamp_utc: str
    timestamp_eastern: str
    user_input_hash: str          # SHA-256 of user input (privacy-safe)
    user_input_preview: str       # First 100 chars
    entropy_class: str            # Detected entropy type
    invariant: str                # What must remain true
    lattice_state: str            # State ? Cause ? Direction
    kappa_score: float            # Coherence estimate
    tau_score: float              # Temporal efficiency estimate
    sigma_score: float            # Risk estimate
    response_preview: str         # First 200 chars of response
    response_hash: str            # SHA-256 of full response
    mode: str                     # Which personality mode was active
    duration_ms: float            # Processing time
    model: str                    # Which LLM was used
    version: str = "1.0"


class ReasoningTraceLogger:
    """
    Logs every reasoning cycle with timestamps and hashes.
    Output is append-only JSONL — suitable for audit, legal, professional use.
    """

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(os.environ.get(
                "AUREON_TRACE_DIR",
                # Detect standard Aureon installation paths
                next((p for p in [
                    r"C:\AUREON_AUTONOMOUS\AUREON_TRACES",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "AUREON_TRACES"),
                    os.path.expanduser("~/AUREON_TRACES"),
                ] if os.path.isdir(os.path.dirname(p))), os.path.expanduser("~/AUREON_TRACES"))
            ))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._cycle_counter = 0
        self._time = TimeOrgan()

    def _log_path(self) -> Path:
        """One file per day."""
        date_str = self._time.now().strftime("%Y-%m-%d")
        return self._log_dir / f"aureon_trace_{date_str}.jsonl"

    def log_cycle(
        self,
        user_input: str,
        response: str,
        *,
        entropy_class: str = "unclassified",
        invariant: str = "",
        lattice_state: str = "",
        kappa: float = 0.0,
        tau: float = 0.0,
        sigma: float = 0.0,
        mode: str = "standard",
        duration_ms: float = 0.0,
        model: str = "gpt-4o-mini",
    ) -> ReasoningTraceEntry:
        """Log a complete reasoning cycle. Returns the entry."""
        self._cycle_counter += 1
        now_utc = datetime.now(timezone.utc)
        now_eastern = self._time.now()

        entry = ReasoningTraceEntry(
            cycle_id=f"AUR-{now_eastern.strftime('%Y%m%d')}-{self._cycle_counter:06d}",
            timestamp_utc=now_utc.isoformat(),
            timestamp_eastern=now_eastern.isoformat(),
            user_input_hash=hashlib.sha256(user_input.encode()).hexdigest(),
            user_input_preview=user_input[:100],
            entropy_class=entropy_class,
            invariant=invariant,
            lattice_state=lattice_state,
            kappa_score=round(kappa, 4),
            tau_score=round(tau, 4),
            sigma_score=round(sigma, 4),
            response_preview=response[:200],
            response_hash=hashlib.sha256(response.encode()).hexdigest(),
            mode=mode,
            duration_ms=round(duration_ms, 2),
            model=model,
        )

        # Append to daily log file
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"   [WARN] Trace log error: {e}")

        return entry

    def get_today_traces(self) -> List[Dict[str, Any]]:
        """Read all traces from today."""
        path = self._log_path()
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def trace_count_today(self) -> int:
        return len(self.get_today_traces())


# ??????????????????????????????????????????????
# BROWSER ORGAN STUBS (Organs 4-8)
# ??????????????????????????????????????????????

class BrowserOrganStub:
    """Placeholder for organs requiring live browser automation."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.status = "awaiting_browser"

    def query(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "organ": self.name,
            "status": self.status,
            "url": self.url,
            "note": f"{self.name} ready — activate when browser hands are operational",
        }


# Organ instances for browser-dependent systems
GeologyOrgan = lambda: BrowserOrganStub(
    "Geology", "https://www.geologyontario.mndm.gov.on.ca/ogsearth.html")
EarthViewOrgan = lambda: BrowserOrganStub(
    "EarthView", "https://earth.google.com")
MapsOrgan = lambda: BrowserOrganStub(
    "Maps", "https://maps.google.com")
WindFieldOrgan = lambda: BrowserOrganStub(
    "WindField", "https://www.windy.com/?42.755,-81.182,5")
CosmicOrgan = lambda: BrowserOrganStub(
    "Cosmic", "https://science.nasa.gov/eyes/")


# ??????????????????????????????????????????????
# UNIFIED ORGAN SYSTEM
# ??????????????????????????????????????????????

class AureonExternalOrgans:
    """
    Unified access to all 14 external organs.
    Instantiate once at boot. Runs silently.
    """

    def __init__(self, hands=None):
        # Live organs
        self.time = TimeOrgan()
        self.calendar = CalendarOrgan()
        self.weather = WeatherOrgan()
        self.trace = ReasoningTraceLogger()

        # Browser-dependent stubs
        self.geology = GeologyOrgan()
        self.earth_view = EarthViewOrgan()
        self.maps = MapsOrgan()
        self.wind_field = WindFieldOrgan()
        self.cosmic = CosmicOrgan()

        # Internal organs (references — actual engines in foundation files)
        self._hands = hands

    def boot_status(self) -> Dict[str, str]:
        """Report organ status at startup."""
        return {
            "time": "[OK] LIVE — " + self.time.now_human(),
            "calendar": "[OK] LIVE — " + json.dumps(self.calendar.today()),
            "weather": "? Ready (needs web fetch)",
            "trace_logger": f"[OK] LIVE — logging to {self.trace._log_dir}",
            "geology": "? Stub (needs browser)",
            "earth_view": "? Stub (needs browser)",
            "maps": "? Stub (needs browser)",
            "wind_field": "? Stub (needs browser)",
            "cosmic": "? Stub (needs browser)",
            "human_field": "? Internal (sensorial binding)",
            "earth_crystal": "? Internal (CSL engine)",
            "semantic": "? Internal (Luméren repo)",
            "coherence": "? Internal (Quantara repo)",
            "energy": "? Internal (AEI repo)",
            "photonic": "? Internal (PCM module)",
        }

    def context_block(self) -> str:
        """
        Generate a compact context string to inject into every system prompt.
        This gives Aureon live awareness of time, date, and calendar position.
        """
        t = self.time.now()
        cal = self.calendar.today()
        tz_name = "EDT" if t.utcoffset() == timedelta(hours=-4) else "EST"

        return (
            f"CURRENT TIME: {t.strftime('%I:%M %p')} {tz_name}, "
            f"{cal['day_of_week']}, {cal['month']} {t.day}, {cal['year']} | "
            f"Season: {cal['season']} | Week {cal['week_number']} | "
            f"Q{cal['quarter']} | Day {cal['day_of_year']} of year\n"
            f"Yesterday: {self.calendar.yesterday()} | "
            f"Tomorrow: {self.calendar.tomorrow()}"
        )


# ??????????????????????????????????????????????
# BOOT FUNCTION
# ??????????????????????????????????????????????

def boot_organs(hands=None, verbose: bool = True) -> AureonExternalOrgans:
    """Boot all external organs. Call once at startup."""
    organs = AureonExternalOrgans(hands=hands)
    if verbose:
        print("\n[EARTH] AUREON EXTERNAL ORGANS — BOOT STATUS")
        print("=" * 50)
        for organ, status in organs.boot_status().items():
            print(f"   {organ:15s} {status}")
        print("=" * 50)
        print(f"   Context: {organs.context_block()}")
        print()
    return organs


# ??????????????????????????????????????????????
# SELF-TEST
# ??????????????????????????????????????????????

if __name__ == "__main__":
    organs = boot_organs()

    # Test trace logging
    entry = organs.trace.log_cycle(
        user_input="Test message for trace validation",
        response="Test response — system operational",
        entropy_class="NullVariant",
        invariant="System integrity",
        kappa=0.92,
        tau=0.88,
        sigma=0.95,
        mode="standard",
        duration_ms=150.0,
    )
    print(f"   Trace logged: {entry.cycle_id}")
    print(f"   Trace file: {organs.trace._log_path()}")
    print(f"   Traces today: {organs.trace.trace_count_today()}")
=======
#!/usr/bin/env python3
"""
Aureon External Organs — Live World Awareness Layer
====================================================
Provides Aureon with continuous, silent access to:

  Organ 1:  TIME        — Exact current time (Toronto / Eastern)
  Organ 2:  CALENDAR    — Full date, day-of-week, week number, season
  Organ 3:  WEATHER     — Real-time local weather (via web fetch)
  Organ 4:  GEOLOGY     — Stub (requires browser)
  Organ 5:  EARTH VIEW  — Stub (requires browser)
  Organ 6:  MAPS        — Stub (requires browser)
  Organ 7:  WIND FIELD  — Stub (requires browser)
  Organ 8:  COSMIC      — Stub (requires browser)
  Organ 9:  HUMAN FIELD — Internal (sensorial binding)
  Organ 10: EARTH CRYSTAL — Internal (CSL engine)
  Organ 11: SEMANTIC    — Internal (Luméren repo)
  Organ 12: COHERENCE   — Internal (Quantara repo)
  Organ 13: ENERGY      — Internal (AEI repo)
  Organ 14: PHOTONIC    — Internal (PCM module)

  TRACE:    Reasoning Trace Logger — timestamps every reasoning cycle
            for court-admissible, professional-grade records.

All organs run silently. No user-facing output unless requested.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import hashlib


# ??????????????????????????????????????????????
# TIMEZONE: Toronto (Eastern)
# ??????????????????????????????????????????????

def _eastern_now() -> datetime:
    """Return current datetime in Eastern Time (EST/EDT aware)."""
    utc_now = datetime.now(timezone.utc)
    # Approximate EST/EDT: EDT = UTC-4 (Mar-Nov), EST = UTC-5 (Nov-Mar)
    # For precise DST, we check month+day heuristically
    month = utc_now.month
    if 3 < month < 11:
        offset = timedelta(hours=-4)  # EDT
    elif month == 3:
        # EDT starts 2nd Sunday of March
        offset = timedelta(hours=-4) if utc_now.day >= 9 else timedelta(hours=-5)
    elif month == 11:
        # EST starts 1st Sunday of November
        offset = timedelta(hours=-5) if utc_now.day >= 3 else timedelta(hours=-4)
    else:
        offset = timedelta(hours=-5)  # EST
    return utc_now.astimezone(timezone(offset))


# ??????????????????????????????????????????????
# ORGAN 1: TIME
# ??????????????????????????????????????????????

class TimeOrgan:
    """Live clock — Toronto Eastern Time."""

    def now(self) -> datetime:
        return _eastern_now()

    def now_iso(self) -> str:
        return self.now().isoformat()

    def now_human(self) -> str:
        n = self.now()
        tz_name = "EDT" if n.utcoffset() == timedelta(hours=-4) else "EST"
        return n.strftime(f"%A, %B %d, %Y at %I:%M:%S %p {tz_name}")

    def now_compact(self) -> str:
        return self.now().strftime("%Y-%m-%d %H:%M:%S %Z")


# ??????????????????????????????????????????????
# ORGAN 2: CALENDAR
# ??????????????????????????????????????????????

class CalendarOrgan:
    """Date awareness — day of week, week number, season, relative dates."""

    def __init__(self):
        self._time = TimeOrgan()

    def today(self) -> Dict[str, Any]:
        n = self._time.now()
        return {
            "date": n.strftime("%Y-%m-%d"),
            "day_of_week": n.strftime("%A"),
            "week_number": int(n.strftime("%W")),
            "month": n.strftime("%B"),
            "year": n.year,
            "season": self._season(n.month),
            "day_of_year": n.timetuple().tm_yday,
            "iso_week": n.isocalendar()[1],
            "quarter": (n.month - 1) // 3 + 1,
        }

    def relative_day(self, offset_days: int) -> str:
        """'yesterday' = -1, 'tomorrow' = +1, 'next Thursday' etc."""
        target = self._time.now() + timedelta(days=offset_days)
        return target.strftime("%A, %B %d, %Y")

    def yesterday(self) -> str:
        return self.relative_day(-1)

    def tomorrow(self) -> str:
        return self.relative_day(1)

    def next_weekday(self, weekday_name: str) -> str:
        """Get date of next occurrence of a weekday (e.g. 'Thursday')."""
        days = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
        target_idx = None
        for i, d in enumerate(days):
            if d.startswith(weekday_name.lower()[:3]):
                target_idx = i
                break
        if target_idx is None:
            return f"Unknown day: {weekday_name}"
        n = self._time.now()
        current_idx = n.weekday()
        delta = (target_idx - current_idx) % 7
        if delta == 0:
            delta = 7  # Next week's occurrence
        return self.relative_day(delta)

    @staticmethod
    def _season(month: int) -> str:
        if month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        elif month in (9, 10, 11):
            return "Autumn"
        else:
            return "Winter"


# ??????????????????????????????????????????????
# ORGAN 3: WEATHER (via web fetch when available)
# ??????????????????????????????????????????????

class WeatherOrgan:
    """Weather awareness — fetches from web or returns cached/stub."""

    def __init__(self, default_location: str = "St. Thomas, Ontario"):
        self.location = default_location
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 1800  # 30 minutes

    def get_weather(self, hands=None) -> Dict[str, Any]:
        """Try to fetch live weather. Falls back to stub if no browser."""
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        # Try fetching via hands (browser) if available
        if hands and hasattr(hands, 'fetch_url_text'):
            try:
                url = f"https://wttr.in/{self.location.replace(' ', '+')}?format=j1"
                raw = hands.fetch_url_text(url)
                if raw:
                    data = json.loads(raw)
                    current = data.get("current_condition", [{}])[0]
                    result = {
                        "location": self.location,
                        "temp_c": current.get("temp_C", "?"),
                        "temp_f": current.get("temp_F", "?"),
                        "feels_like_c": current.get("FeelsLikeC", "?"),
                        "humidity": current.get("humidity", "?"),
                        "description": current.get("weatherDesc", [{}])[0].get("value", "?"),
                        "wind_kmh": current.get("windspeedKmph", "?"),
                        "wind_dir": current.get("winddir16Point", "?"),
                        "visibility_km": current.get("visibility", "?"),
                        "pressure_mb": current.get("pressure", "?"),
                        "source": "wttr.in",
                        "fetched_at": TimeOrgan().now_iso(),
                    }
                    self._cache = result
                    self._cache_time = now
                    return result
            except Exception:
                pass

        # Stub response when no live data available
        return {
            "location": self.location,
            "status": "awaiting_live_connection",
            "note": "Weather organ ready — needs web access to activate",
        }


# ??????????????????????????????????????????????
# REASONING TRACE LOGGER
# ??????????????????????????????????????????????

@dataclass
class ReasoningTraceEntry:
    """A single reasoning cycle record — court-admissible format."""
    cycle_id: str
    timestamp_utc: str
    timestamp_eastern: str
    user_input_hash: str          # SHA-256 of user input (privacy-safe)
    user_input_preview: str       # First 100 chars
    entropy_class: str            # Detected entropy type
    invariant: str                # What must remain true
    lattice_state: str            # State ? Cause ? Direction
    kappa_score: float            # Coherence estimate
    tau_score: float              # Temporal efficiency estimate
    sigma_score: float            # Risk estimate
    response_preview: str         # First 200 chars of response
    response_hash: str            # SHA-256 of full response
    mode: str                     # Which personality mode was active
    duration_ms: float            # Processing time
    model: str                    # Which LLM was used
    version: str = "1.0"


class ReasoningTraceLogger:
    """
    Logs every reasoning cycle with timestamps and hashes.
    Output is append-only JSONL — suitable for audit, legal, professional use.
    """

    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path(os.environ.get(
                "AUREON_TRACE_DIR",
                # Detect standard Aureon installation paths
                next((p for p in [
                    r"C:\AUREON_AUTONOMOUS\AUREON_TRACES",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "AUREON_TRACES"),
                    os.path.expanduser("~/AUREON_TRACES"),
                ] if os.path.isdir(os.path.dirname(p))), os.path.expanduser("~/AUREON_TRACES"))
            ))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._cycle_counter = 0
        self._time = TimeOrgan()

    def _log_path(self) -> Path:
        """One file per day."""
        date_str = self._time.now().strftime("%Y-%m-%d")
        return self._log_dir / f"aureon_trace_{date_str}.jsonl"

    def log_cycle(
        self,
        user_input: str,
        response: str,
        *,
        entropy_class: str = "unclassified",
        invariant: str = "",
        lattice_state: str = "",
        kappa: float = 0.0,
        tau: float = 0.0,
        sigma: float = 0.0,
        mode: str = "standard",
        duration_ms: float = 0.0,
        model: str = "gpt-4o-mini",
    ) -> ReasoningTraceEntry:
        """Log a complete reasoning cycle. Returns the entry."""
        self._cycle_counter += 1
        now_utc = datetime.now(timezone.utc)
        now_eastern = self._time.now()

        entry = ReasoningTraceEntry(
            cycle_id=f"AUR-{now_eastern.strftime('%Y%m%d')}-{self._cycle_counter:06d}",
            timestamp_utc=now_utc.isoformat(),
            timestamp_eastern=now_eastern.isoformat(),
            user_input_hash=hashlib.sha256(user_input.encode()).hexdigest(),
            user_input_preview=user_input[:100],
            entropy_class=entropy_class,
            invariant=invariant,
            lattice_state=lattice_state,
            kappa_score=round(kappa, 4),
            tau_score=round(tau, 4),
            sigma_score=round(sigma, 4),
            response_preview=response[:200],
            response_hash=hashlib.sha256(response.encode()).hexdigest(),
            mode=mode,
            duration_ms=round(duration_ms, 2),
            model=model,
        )

        # Append to daily log file
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"   [WARN] Trace log error: {e}")

        return entry

    def get_today_traces(self) -> List[Dict[str, Any]]:
        """Read all traces from today."""
        path = self._log_path()
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def trace_count_today(self) -> int:
        return len(self.get_today_traces())


# ??????????????????????????????????????????????
# BROWSER ORGAN STUBS (Organs 4-8)
# ??????????????????????????????????????????????

class BrowserOrganStub:
    """Placeholder for organs requiring live browser automation."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.status = "awaiting_browser"

    def query(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "organ": self.name,
            "status": self.status,
            "url": self.url,
            "note": f"{self.name} ready — activate when browser hands are operational",
        }


# Organ instances for browser-dependent systems
GeologyOrgan = lambda: BrowserOrganStub(
    "Geology", "https://www.geologyontario.mndm.gov.on.ca/ogsearth.html")
EarthViewOrgan = lambda: BrowserOrganStub(
    "EarthView", "https://earth.google.com")
MapsOrgan = lambda: BrowserOrganStub(
    "Maps", "https://maps.google.com")
WindFieldOrgan = lambda: BrowserOrganStub(
    "WindField", "https://www.windy.com/?42.755,-81.182,5")
CosmicOrgan = lambda: BrowserOrganStub(
    "Cosmic", "https://science.nasa.gov/eyes/")


# ??????????????????????????????????????????????
# UNIFIED ORGAN SYSTEM
# ??????????????????????????????????????????????

class AureonExternalOrgans:
    """
    Unified access to all 14 external organs.
    Instantiate once at boot. Runs silently.
    """

    def __init__(self, hands=None):
        # Live organs
        self.time = TimeOrgan()
        self.calendar = CalendarOrgan()
        self.weather = WeatherOrgan()
        self.trace = ReasoningTraceLogger()

        # Browser-dependent stubs
        self.geology = GeologyOrgan()
        self.earth_view = EarthViewOrgan()
        self.maps = MapsOrgan()
        self.wind_field = WindFieldOrgan()
        self.cosmic = CosmicOrgan()

        # Internal organs (references — actual engines in foundation files)
        self._hands = hands

    def boot_status(self) -> Dict[str, str]:
        """Report organ status at startup."""
        return {
            "time": "[OK] LIVE — " + self.time.now_human(),
            "calendar": "[OK] LIVE — " + json.dumps(self.calendar.today()),
            "weather": "? Ready (needs web fetch)",
            "trace_logger": f"[OK] LIVE — logging to {self.trace._log_dir}",
            "geology": "? Stub (needs browser)",
            "earth_view": "? Stub (needs browser)",
            "maps": "? Stub (needs browser)",
            "wind_field": "? Stub (needs browser)",
            "cosmic": "? Stub (needs browser)",
            "human_field": "? Internal (sensorial binding)",
            "earth_crystal": "? Internal (CSL engine)",
            "semantic": "? Internal (Luméren repo)",
            "coherence": "? Internal (Quantara repo)",
            "energy": "? Internal (AEI repo)",
            "photonic": "? Internal (PCM module)",
        }

    def context_block(self) -> str:
        """
        Generate a compact context string to inject into every system prompt.
        This gives Aureon live awareness of time, date, and calendar position.
        """
        t = self.time.now()
        cal = self.calendar.today()
        tz_name = "EDT" if t.utcoffset() == timedelta(hours=-4) else "EST"

        return (
            f"CURRENT TIME: {t.strftime('%I:%M %p')} {tz_name}, "
            f"{cal['day_of_week']}, {cal['month']} {t.day}, {cal['year']} | "
            f"Season: {cal['season']} | Week {cal['week_number']} | "
            f"Q{cal['quarter']} | Day {cal['day_of_year']} of year\n"
            f"Yesterday: {self.calendar.yesterday()} | "
            f"Tomorrow: {self.calendar.tomorrow()}"
        )


# ??????????????????????????????????????????????
# BOOT FUNCTION
# ??????????????????????????????????????????????

def boot_organs(hands=None, verbose: bool = True) -> AureonExternalOrgans:
    """Boot all external organs. Call once at startup."""
    organs = AureonExternalOrgans(hands=hands)
    if verbose:
        print("\n[EARTH] AUREON EXTERNAL ORGANS — BOOT STATUS")
        print("=" * 50)
        for organ, status in organs.boot_status().items():
            print(f"   {organ:15s} {status}")
        print("=" * 50)
        print(f"   Context: {organs.context_block()}")
        print()
    return organs


# ??????????????????????????????????????????????
# SELF-TEST
# ??????????????????????????????????????????????

if __name__ == "__main__":
    organs = boot_organs()

    # Test trace logging
    entry = organs.trace.log_cycle(
        user_input="Test message for trace validation",
        response="Test response — system operational",
        entropy_class="NullVariant",
        invariant="System integrity",
        kappa=0.92,
        tau=0.88,
        sigma=0.95,
        mode="standard",
        duration_ms=150.0,
    )
    print(f"   Trace logged: {entry.cycle_id}")
    print(f"   Trace file: {organs.trace._log_path()}")
    print(f"   Traces today: {organs.trace.trace_count_today()}")
>>>>>>> 3e64d789316217ee128862f1ededbace704ec132
