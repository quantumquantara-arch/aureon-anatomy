"""
AUREON EARS - System Audio Capture + Speech Recognition
Captures what is playing on the laptop (system audio loopback).
Uses Windows WASAPI loopback via PowerShell, then transcribes with whisper.

DEPENDENCIES (auto-installed on first use):
  pip install sounddevice numpy
  pip install openai-whisper  (optional, for transcription)
"""

import subprocess
import os
import json
import time
import tempfile
import threading
from typing import Optional, Dict, Any, List

class AureonEars:
    """Captures system audio and transcribes speech."""
    
    def __init__(self, capture_dir: Optional[str] = None):
        self.capture_dir = capture_dir or os.path.join(
            os.environ.get("USERPROFILE", os.path.expanduser("~")),
            "AUREON_TRACES", "audio_captures"
        )
        os.makedirs(self.capture_dir, exist_ok=True)
        
        self._is_listening = False
        self._listen_thread = None
        self._last_transcript = ""
        self._last_audio_info = {}
        self._whisper_model = None
        self._has_sounddevice = False
        self._has_whisper = False
        
        # Check dependencies on init
        self._check_deps()
    
    def _check_deps(self):
        """Check what audio libraries are available."""
        try:
            import sounddevice
            self._has_sounddevice = True
        except ImportError:
            self._has_sounddevice = False
        
        try:
            import whisper
            self._has_whisper = True
        except ImportError:
            self._has_whisper = False
    
    def install_deps(self) -> str:
        """Install audio dependencies. Returns status."""
        results = []
        try:
            r = subprocess.run(
                ["pip", "install", "sounddevice", "numpy", "--break-system-packages", "-q"],
                capture_output=True, text=True, timeout=120
            )
            results.append(f"sounddevice: {'OK' if r.returncode == 0 else r.stderr[:200]}")
        except Exception as e:
            results.append(f"sounddevice: {e}")
        
        try:
            r = subprocess.run(
                ["pip", "install", "openai-whisper", "--break-system-packages", "-q"],
                capture_output=True, text=True, timeout=300
            )
            results.append(f"whisper: {'OK' if r.returncode == 0 else r.stderr[:200]}")
        except Exception as e:
            results.append(f"whisper: {e}")
        
        self._check_deps()
        return "; ".join(results)
    
    # ========================================================
    # METHOD 1: PowerShell audio capture (always works on Windows)
    # ========================================================
    
    def get_now_playing(self) -> Dict[str, Any]:
        """Get info about what's currently playing on Windows.
        Uses PowerShell to check active audio sessions."""
        try:
            # Check what audio processes are active
            ps_cmd = """
            $sessions = Get-Process | Where-Object {
                $_.MainWindowTitle -ne '' -and (
                    $_.ProcessName -match 'chrome|firefox|edge|msedge|spotify|vlc|wmplayer|foobar|musicbee|winamp|audacity|brave|opera'
                )
            } | Select-Object ProcessName, MainWindowTitle, Id
            $sessions | ConvertTo-Json -Compress
            """
            r = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=10
            )
            
            audio_apps = []
            if r.returncode == 0 and r.stdout.strip():
                try:
                    data = json.loads(r.stdout.strip())
                    if isinstance(data, dict):
                        data = [data]
                    audio_apps = data
                except json.JSONDecodeError:
                    pass
            
            # Also check for audio sessions using audiodg
            ps_audio = """
            $playing = Get-Process audiodg -ErrorAction SilentlyContinue
            if ($playing) { "audiodg_active" } else { "audiodg_inactive" }
            """
            r2 = subprocess.run(
                ["powershell", "-Command", ps_audio],
                capture_output=True, text=True, timeout=5
            )
            audiodg = r2.stdout.strip() if r2.returncode == 0 else "unknown"
            
            # Check Spotify specifically (common podcast source)
            spotify_info = self._get_spotify_info()
            
            # Check browser tabs for known podcast/media sites
            browser_info = self._get_browser_media_info()
            
            result = {
                "audio_engine": audiodg,
                "audio_apps": audio_apps,
                "spotify": spotify_info,
                "browser_media": browser_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._last_audio_info = result
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_spotify_info(self) -> Dict[str, Any]:
        """Get what Spotify is playing (if running)."""
        try:
            ps_cmd = """
            $spotify = Get-Process spotify -ErrorAction SilentlyContinue | 
                Where-Object { $_.MainWindowTitle -ne '' } |
                Select-Object -First 1
            if ($spotify) {
                @{ running = $true; title = $spotify.MainWindowTitle } | ConvertTo-Json -Compress
            } else {
                @{ running = $false; title = '' } | ConvertTo-Json -Compress
            }
            """
            r = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout.strip())
        except:
            pass
        return {"running": False, "title": ""}
    
    def _get_browser_media_info(self) -> Dict[str, Any]:
        """Check if browser has media playing (Chrome/Edge window titles often show media)."""
        try:
            ps_cmd = """
            Get-Process chrome, msedge, firefox, brave -ErrorAction SilentlyContinue |
                Where-Object { $_.MainWindowTitle -ne '' } |
                Select-Object ProcessName, MainWindowTitle |
                ConvertTo-Json -Compress
            """
            r = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and r.stdout.strip():
                data = json.loads(r.stdout.strip())
                if isinstance(data, dict):
                    data = [data]
                # Look for media indicators in titles
                media_tabs = []
                for tab in data:
                    title = tab.get("MainWindowTitle", "")
                    if any(kw in title.lower() for kw in [
                        "youtube", "spotify", "podcast", "playing", "soundcloud",
                        "apple podcasts", "overcast", "pocket casts", "audible",
                        "bandcamp", "tidal", "deezer", "pandora", "tunein",
                        "stitcher", "castbox", "radio", "music", "listen"
                    ]):
                        media_tabs.append(tab)
                return {"tabs": data, "media_tabs": media_tabs}
        except:
            pass
        return {"tabs": [], "media_tabs": []}
    
    # ========================================================
    # METHOD 2: WASAPI loopback capture (requires sounddevice)
    # ========================================================
    
    def capture_audio(self, duration_seconds: int = 10, filename: Optional[str] = None) -> Dict[str, Any]:
        """Capture system audio for given duration using WASAPI loopback.
        Returns path to WAV file."""
        if not self._has_sounddevice:
            return {"error": "sounddevice not installed. Call install_deps() first.",
                    "fallback": self.capture_audio_powershell(duration_seconds, filename)}
        
        try:
            import sounddevice as sd
            import numpy as np
            
            if filename is None:
                filename = os.path.join(
                    self.capture_dir,
                    f"capture_{int(time.time())}.wav"
                )
            
            # Find WASAPI loopback device
            devices = sd.query_devices()
            loopback_device = None
            for i, dev in enumerate(devices):
                name = dev.get('name', '').lower()
                if 'loopback' in name or ('stereo mix' in name):
                    loopback_device = i
                    break
            
            if loopback_device is None:
                # Try default output device as input (some drivers support this)
                return {"error": "No loopback device found. Enable 'Stereo Mix' in Windows Sound settings.",
                        "fallback": self.capture_audio_powershell(duration_seconds, filename)}
            
            samplerate = int(devices[loopback_device]['default_samplerate'])
            channels = min(2, int(devices[loopback_device]['max_input_channels']))
            
            print(f"   [EAR] Capturing {duration_seconds}s from device {loopback_device}...")
            recording = sd.rec(
                int(duration_seconds * samplerate),
                samplerate=samplerate,
                channels=channels,
                device=loopback_device,
                dtype='float32'
            )
            sd.wait()
            
            # Save as WAV
            import wave
            with wave.open(filename, 'w') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(samplerate)
                wf.writeframes((recording * 32767).astype(np.int16).tobytes())
            
            # Check if audio contains actual sound (not silence)
            rms = np.sqrt(np.mean(recording ** 2))
            has_audio = rms > 0.001
            
            return {
                "file": filename,
                "duration": duration_seconds,
                "samplerate": samplerate,
                "channels": channels,
                "rms": float(rms),
                "has_audio": has_audio,
                "silence": not has_audio,
            }
            
        except Exception as e:
            return {"error": str(e),
                    "fallback": self.capture_audio_powershell(duration_seconds, filename)}
    
    def capture_audio_powershell(self, duration_seconds: int = 10, filename: Optional[str] = None) -> Dict[str, Any]:
        """Fallback: capture audio using PowerShell and ffmpeg if available."""
        if filename is None:
            filename = os.path.join(self.capture_dir, f"capture_{int(time.time())}.wav")
        
        try:
            # Try ffmpeg with dshow (DirectShow) audio capture
            ps_cmd = f"""
            $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
            if ($ffmpeg) {{
                & ffmpeg -f dshow -i audio="Stereo Mix" -t {duration_seconds} -y "{filename}" 2>&1 | Out-Null
                if (Test-Path "{filename}") {{ "captured" }} else {{ "no_stereo_mix" }}
            }} else {{
                "no_ffmpeg"
            }}
            """
            r = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=duration_seconds + 15
            )
            status = r.stdout.strip()
            if status == "captured" and os.path.exists(filename):
                return {"file": filename, "method": "ffmpeg_dshow", "duration": duration_seconds}
            else:
                return {"error": f"PowerShell capture failed: {status}",
                        "hint": "Enable 'Stereo Mix' in Sound Settings > Recording Devices"}
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================
    # METHOD 3: Whisper transcription
    # ========================================================
    
    def transcribe(self, audio_file: str, language: str = "en") -> Dict[str, Any]:
        """Transcribe audio file using OpenAI Whisper (local)."""
        if not self._has_whisper:
            return {"error": "whisper not installed. Call install_deps() first."}
        
        try:
            import whisper
            
            if self._whisper_model is None:
                print("   [EAR] Loading Whisper model (first time)...")
                self._whisper_model = whisper.load_model("base")
            
            result = self._whisper_model.transcribe(audio_file, language=language)
            
            self._last_transcript = result.get("text", "")
            return {
                "text": self._last_transcript,
                "language": result.get("language", language),
                "segments": len(result.get("segments", [])),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def listen_and_transcribe(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """Capture audio and transcribe in one step."""
        capture = self.capture_audio(duration_seconds)
        if "error" in capture and "file" not in capture:
            # Try fallback
            if "fallback" in capture and isinstance(capture["fallback"], dict):
                capture = capture["fallback"]
            if "error" in capture and "file" not in capture:
                return capture
        
        audio_file = capture.get("file")
        if not audio_file or not os.path.exists(audio_file):
            return {"error": "No audio file captured", "capture_result": capture}
        
        if capture.get("silence"):
            return {"text": "[SILENCE - no audio detected]", "capture": capture}
        
        transcript = self.transcribe(audio_file)
        transcript["capture"] = capture
        return transcript
    
    # ========================================================
    # STATUS
    # ========================================================
    
    def status(self) -> Dict[str, Any]:
        """Return current ear status."""
        now_playing = self.get_now_playing()
        return {
            "operational": True,
            "has_sounddevice": self._has_sounddevice,
            "has_whisper": self._has_whisper,
            "can_capture_loopback": self._has_sounddevice,
            "can_transcribe": self._has_whisper,
            "now_playing": now_playing,
            "last_transcript": self._last_transcript[:200] if self._last_transcript else "",
            "capture_dir": self.capture_dir,
        }
    
    def get_honest_answer(self) -> str:
        """When asked 'what is playing', give an HONEST answer based on actual data."""
        info = self.get_now_playing()
        
        parts = []
        
        # Spotify
        spotify = info.get("spotify", {})
        if spotify.get("running") and spotify.get("title"):
            title = spotify["title"]
            if title.lower() not in ["spotify", "spotify free", "spotify premium"]:
                parts.append(f"Spotify is playing: {title}")
        
        # Browser media
        browser = info.get("browser_media", {})
        media_tabs = browser.get("media_tabs", [])
        if media_tabs:
            for tab in media_tabs:
                parts.append(f"Browser tab: {tab.get('MainWindowTitle', 'unknown')}")
        
        # Other audio apps
        for app in info.get("audio_apps", []):
            name = app.get("ProcessName", "")
            title = app.get("MainWindowTitle", "")
            if name.lower() not in ["chrome", "msedge", "firefox", "brave", "spotify"]:
                parts.append(f"{name}: {title}")
        
        if parts:
            return "Currently playing:\n" + "\n".join(f"  - {p}" for p in parts)
        
        # Check if audiodg is active (something is making sound)
        if info.get("audio_engine") == "audiodg_active":
            # Audio engine active but can't identify source
            all_tabs = browser.get("tabs", [])
            if all_tabs:
                tab_titles = [t.get("MainWindowTitle", "") for t in all_tabs if t.get("MainWindowTitle")]
                return ("Audio engine is active (something is playing) but I cannot identify the exact source. "
                        f"Open browser tabs: {', '.join(tab_titles[:5])}")
            return "Audio engine is active but I cannot identify what is playing."
        
        return "No audio sources detected. Nothing appears to be playing."


# Quick test
if __name__ == "__main__":
    ears = AureonEars()
    print("=== AUREON EARS STATUS ===")
    s = ears.status()
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("\n=== NOW PLAYING ===")
    print(ears.get_honest_answer())
