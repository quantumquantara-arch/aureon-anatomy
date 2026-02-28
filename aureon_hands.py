<<<<<<< HEAD
import os
import subprocess
import time
from typing import Optional, List
import psutil
import pyttsx3
import speech_recognition as sr
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
"""
AUREON HANDS - COMPLETE
========================
Two capabilities that work INDEPENDENTLY:

1. FILE OPERATIONS (always work, no browser needed):
   - search_files, read_file, write_file, list_files
   - run_command (PowerShell/cmd)
   
2. BROWSER OPERATIONS (need Chrome with --remote-debugging-port=9222):
   - click_on_text, type_text, press, go_to_url, switch_tab, scroll, new_tab
   - Auto-retries connection if browser wasn't ready at startup
"""

from typing import Dict, Any, Optional, List
import subprocess
import time
import os
from pathlib import Path

try:
    from aureon_surgeon import AureonSurgeon
except ImportError:
    AureonSurgeon = None

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class AureonHands:
    """
    AUREON's hands - controls files AND browser.
    File operations ALWAYS work.
    Browser operations auto-retry connection.
    """

    def __init__(self, base_dir: str = r"C:\AUREON_AUTONOMOUS"):
        self.base_dir = Path(base_dir)
        self.driver = None
        self.browser_connected = False
        self._last_connect_attempt = 0
        
        # Initialize the surgical code editor
        self.surgeon = AureonSurgeon(base_dir=base_dir) if AureonSurgeon else None

        if SELENIUM_AVAILABLE:
            self._try_connect_browser()
        else:
            print("\u26A0  Selenium not installed: pip install selenium")
            print("   Browser control disabled. File operations still work.")

    # ??????????????????????????????????????????????????????????
    # BROWSER CONNECTION (auto-retry)
    # ??????????????????????????????????????????????????????????

    def _try_connect_browser(self) -> bool:
        """Try to connect to existing browser. Auto-launches Chrome if needed."""
        now = time.time()
        # Don't spam retries — wait at least 10 seconds between attempts
        if now - self._last_connect_attempt < 10:
            return self.browser_connected
        self._last_connect_attempt = now

        if self.browser_connected and self.driver:
            # Check if still alive
            try:
                _ = self.driver.title
                return True
            except Exception:
                self.browser_connected = False
                self.driver = None

        # Step 1: Check if Chrome is already listening on 9222
        import socket
        chrome_listening = False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(("127.0.0.1", 9222))
            s.close()
            chrome_listening = True
        except Exception:
            pass
        
        # Step 2: If Chrome is NOT listening, launch it ourselves
        if not chrome_listening:
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ]
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ]
            
            launched = False
            for browser_path in chrome_paths + edge_paths:
                if os.path.exists(browser_path):
                    try:
                        browser_name = "Chrome" if "chrome" in browser_path.lower() else "Edge"
                        print(f"   [LAUNCH] Launching {browser_name} with --remote-debugging-port=9222...")
                        subprocess.Popen(
                            [browser_path, "--remote-debugging-port=9222", "--no-first-run"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        # Wait for it to start
                        for attempt in range(15):  # Up to 15 seconds
                            time.sleep(1)
                            try:
                                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                s.settimeout(1)
                                s.connect(("127.0.0.1", 9222))
                                s.close()
                                print(f"   [OK] {browser_name} is now listening on port 9222")
                                launched = True
                                break
                            except Exception:
                                pass
                        if launched:
                            break
                    except Exception as launch_err:
                        print(f"   [WARN] Failed to launch {browser_path}: {launch_err}")
            
            if not launched:
                print("   [FAIL] Could not launch any browser with debug port")
                return False

        # Step 3: Connect via Selenium
        # Try Chrome
        try:
            options = webdriver.ChromeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Chrome(options=options)
            self.browser_connected = True
            print("\u2705 Connected to Chrome (port 9222)")
            return True
        except Exception as chrome_err:
            short_err = str(chrome_err).split('\n')[0][:120]
            print(f"   Chrome selenium failed: {short_err}")

        # Try Edge
        try:
            options = webdriver.EdgeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Edge(options=options)
            self.browser_connected = True
            print("\u2705 Connected to Edge (port 9222)")
            return True
        except Exception as edge_err:
            short_err = str(edge_err).split('\n')[0][:120]
            print(f"   Edge selenium failed: {short_err}")

        if not self.browser_connected:
            print("\u26A0  Browser launched but Selenium could not connect")
            print("    This usually means chromedriver version doesn't match Chrome version")
            print("    Update: pip install --upgrade selenium")
        return False

    def _ensure_browser(self) -> bool:
        """Ensure browser is connected, retry if needed."""
        if self.browser_connected:
            try:
                _ = self.driver.title
                return True
            except Exception:
                self.browser_connected = False
                self.driver = None

        if SELENIUM_AVAILABLE:
            return self._try_connect_browser()
        return False

    # ??????????????????????????????????????????????????????????
    # DISPATCH - routes all operations
    # ??????????????????????????????????????????????????????????

    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """Route operations to handlers"""

        # ?? File operations (ALWAYS work) ????????????
        if op == "search_files":
            return self.search_files(kwargs.get("query", ""), kwargs.get("root", str(self.base_dir)))
        elif op == "read_file":
            return self.read_file(kwargs.get("path", ""))
        elif op == "write_file":
            return self.write_file(kwargs.get("path", ""), kwargs.get("content", ""))
        elif op == "list_files":
            return self.list_files(kwargs.get("path", str(self.base_dir)))
        elif op == "read_directory":
            return self.read_directory(kwargs.get("path", ""), kwargs.get("pattern", "*.md"))
        elif op == "scan_all_files":
            return self.scan_all_files(
                root=kwargs.get("root", kwargs.get("path", str(self.base_dir))),
                extensions=kwargs.get("extensions", ".md,.py,.kernel")
            )
        elif op == "run_command":
            return self.run_command(kwargs.get("command", ""))
        elif op == "google_search":
            return self.google_search(kwargs.get("query", ""))
        elif op == "open_link":
            return self.open_link(kwargs.get("url", ""))
        
        # ?? Surgical code operations ?????
        elif op in ("scan_file", "select_focus", "apply_edit", "verify_syntax", "revert", "mutation_log"):
            if self.surgeon:
                return self.surgeon.dispatch(op, **kwargs)
            else:
                return {"ok": False, "error": "Surgeon not available — install aureon_surgeon.py"}

        # ?? Browser operations (need connection) ?????
        if not self._ensure_browser():
            return {
                "ok": False,
                "error": "browser_not_connected",
                "hint": "Start Chrome with: Start-Process 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' -ArgumentList '--remote-debugging-port=9222'",
            }

        if op == "click_on_text":
            return self.click_on_text(kwargs.get("text", ""))
        elif op == "find_and_click":
            return self.find_and_click(kwargs.get("description", ""))
        elif op == "type_text":
            return self.type_text(kwargs.get("text", ""), kwargs.get("clear_first", False))
        elif op == "press":
            return self.press_key(kwargs.get("key", "enter"))
        elif op == "hotkey":
            return self.hotkey(kwargs.get("keys", []))
        elif op == "click":
            return self.click_active()
        elif op == "go_to_url":
            return self.go_to_url(kwargs.get("url", ""))
        elif op == "switch_tab":
            return self.switch_to_tab(kwargs.get("title", kwargs.get("title_contains", "")))
        elif op == "focus_window":
            return self.switch_to_tab(kwargs.get("title_contains", ""))
        elif op == "scroll":
            amt = kwargs.get("amount", 300)
            direction = kwargs.get("direction", "down" if amt >= 0 else "up")
            return self.scroll(direction, abs(amt))
        elif op == "new_tab":
            return self.new_tab(kwargs.get("url"))
        elif op == "get_page_text":
            return self.get_page_text()
        elif op == "get_page_title":
            return self.get_page_title()
        elif op == "get_tabs":
            return self.get_tabs()
        elif op == "return_to_chat":
            return self.return_to_chat()
        elif op == "find_element":
            return self.find_element(kwargs.get("css", kwargs.get("xpath", "")), kwargs.get("by", "css"))
        elif op == "click_element":
            return self.click_element(kwargs.get("css", kwargs.get("xpath", "")), kwargs.get("by", "css"))
        elif op == "type_into":
            return self.type_into(kwargs.get("css", kwargs.get("xpath", "")), kwargs.get("text", ""), kwargs.get("by", "css"))
        else:
            # ?? Desktop operations (pyautogui — controls ENTIRE screen) ??
            if op == "desktop_click":
                return self.desktop_click(kwargs.get("x", 0), kwargs.get("y", 0))
            elif op == "desktop_move":
                return self.desktop_move(kwargs.get("x", 0), kwargs.get("y", 0))
            elif op == "desktop_type":
                return self.desktop_type(kwargs.get("text", ""))
            elif op == "desktop_hotkey":
                return self.desktop_hotkey(kwargs.get("keys", []))
            elif op == "desktop_screenshot":
                return self.desktop_screenshot()
            elif op == "taskbar_click":
                return self.taskbar_click(kwargs.get("app_name", ""))
            else:
                return {"ok": False, "error": f"unknown_operation: {op}"}

    # ??????????????????????????????????????????????????????????
    def scan_all_files(self, root: str = None, extensions: str = ".md,.py,.kernel") -> Dict[str, Any]:
        """
        Scan a directory and return ALL files matching given extensions.
        Unlike search_files, this doesn't search content - just lists by type.
        
        Args:
            root: Directory to scan (defaults to base_dir)
            extensions: Comma-separated extensions like ".md,.py,.kernel"
        """
        try:
            p = Path(root) if root else self.base_dir
            if not p.exists():
                return {"ok": False, "error": f"Directory not found: {root}"}
            
            ext_list = [e.strip().lower() for e in extensions.split(",")]
            # Ensure dots
            ext_list = [e if e.startswith(".") else f".{e}" for e in ext_list]
            
            skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", 
                        "BROWSER_PROFILE", "driver", "msedgedriver"}
            
            files = []
            for f in p.rglob("*"):
                if f.is_dir():
                    continue
                if any(sd in f.parts for sd in skip_dirs):
                    continue
                if f.suffix.lower() in ext_list:
                    files.append({
                        "path": str(f),
                        "name": f.name,
                        "type": f.suffix.lower(),
                        "size": f.stat().st_size,
                        "relative": str(f.relative_to(p)) if f.is_relative_to(p) else f.name,
                    })
            
            # Sort by type then name
            files.sort(key=lambda x: (x["type"], x["name"].lower()))
            
            # Build summary by type
            type_counts = {}
            for f in files:
                t = f["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
            
            summary_parts = [f"{count} {ext} files" for ext, count in sorted(type_counts.items())]
            summary = ", ".join(summary_parts)
            
            return {
                "ok": True,
                "files": files,
                "count": len(files),
                "by_type": type_counts,
                "output": f"Found {len(files)} files ({summary}) in {p.name}",
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # FILE OPERATIONS (always work, no browser needed)
    # ??????????????????????????????????????????????????????????

    def search_files(self, query: str, root: str = None) -> Dict[str, Any]:
        """Search for files containing text or matching name.
        Prioritizes filename matches over content matches.
        Skips code files (.py) and huge files (>200KB) for content search."""
        try:
            p = Path(root) if root else self.base_dir
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try common variations before giving up
                for candidate in [
                    self.base_dir / "AUREON_FOUNDATION",
                    self.base_dir,
                ]:
                    if candidate.exists():
                        p = candidate
                        break
            filename_matches = []
            content_matches = []
            extensions = (".txt", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg", ".html", ".css", ".js", ".csv", ".log")
            skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", "BROWSER_PROFILE", "driver"}
            # Files that should never be returned as content matches
            skip_content_files = {
                "aureon_brain.py", "aureon_web_interface.py", "aureon_hands.py",
                "aureon_kernel_loader.py", "AUREON_COMPILED_IDENTITY.md",
            }

            for f in p.rglob("*"):
                if len(filename_matches) + len(content_matches) >= 30:
                    break
                if f.is_dir():
                    continue
                if any(sd in f.parts for sd in skip_dirs):
                    continue

                # Match filename first (always include, even .py)
                if query.lower() in f.name.lower():
                    filename_matches.append({"path": str(f), "match": "filename"})
                    continue

                # Skip code files and huge files for content search
                if f.name in skip_content_files:
                    continue
                if f.suffix.lower() == ".py":
                    continue
                if f.stat().st_size > 200000:  # Skip files > 200KB
                    continue

                # Match content for text files
                if f.suffix.lower() in extensions:
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")[:50000]
                        if query.lower() in text.lower():
                            for line in text.split("\n"):
                                if query.lower() in line.lower():
                                    content_matches.append({
                                        "path": str(f),
                                        "match": "content",
                                        "context": line.strip()[:200]
                                    })
                                    break
                    except Exception:
                        continue

            # Filename matches first, then content matches
            matches = filename_matches + content_matches
            return {
                "ok": True,
                "matches": matches,
                "count": len(matches),
                "output": f"Found {len(matches)} files matching '{query}' ({len(filename_matches)} by name, {len(content_matches)} by content)"
            }
        except Exception as e:
            # POWERSHELL FALLBACK: If Python path traversal fails, use PowerShell
            try:
                safe_query = query.replace("'", "''").replace('"', '""')
                # Search by filename first
                ps_cmd = (
                    f'Get-ChildItem -Path "C:\\AUREON_AUTONOMOUS" -Recurse -ErrorAction SilentlyContinue | '
                    f'Where-Object {{ $_.Name -like "*{safe_query.replace(" ", "*")}*" }} | '
                    f'Select-Object -First 15 FullName'
                )
                ps_result = subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    capture_output=True, text=True, timeout=15,
                    cwd=str(self.base_dir)
                )
                if ps_result.returncode == 0 and ps_result.stdout.strip():
                    lines = [l.strip() for l in ps_result.stdout.strip().split('\n') if l.strip() and not l.strip().startswith('-')]
                    # Skip header line
                    paths = [l for l in lines if '\\' in l or '/' in l]
                    if paths:
                        matches = [{"path": p, "match": "powershell_filename"} for p in paths]
                        return {
                            "ok": True,
                            "matches": matches,
                            "count": len(matches),
                            "output": f"PowerShell found {len(matches)} files matching '{query}'"
                        }
            except Exception:
                pass
            return {"ok": False, "error": repr(e)}

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file's contents - supports text files AND PDFs"""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try searching for it by name
                for match in self.base_dir.rglob(f"*{Path(path).name}*"):
                    if match.is_file():
                        p = match
                        break
                else:
                    return {"ok": False, "error": f"File not found: {path}"}

            # Handle PDFs
            if p.suffix.lower() == ".pdf":
                return self._read_pdf(p)

            content = p.read_text(encoding="utf-8", errors="ignore")
            return {
                "ok": True,
                "path": str(p),
                "content": content[:100000],
                "size": len(content),
                "output": f"Read {len(content)} chars from {p.name}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def _read_pdf(self, p: Path) -> Dict[str, Any]:
        """Extract text from a PDF file"""
        # Try multiple PDF libraries
        text = ""

        # Method 1: PyPDF2 / pypdf
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader

            reader = PdfReader(str(p))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"--- Page {i+1} ---\n{page_text}")
            text = "\n\n".join(pages)
            if text.strip():
                return {
                    "ok": True,
                    "path": str(p),
                    "content": text[:100000],
                    "pages": len(reader.pages),
                    "size": len(text),
                    "output": f"Read PDF: {p.name} ({len(reader.pages)} pages, {len(text)} chars)"
                }
        except ImportError:
            pass
        except Exception:
            pass

        # Method 2: pdfminer
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(str(p))
            if text.strip():
                return {
                    "ok": True,
                    "path": str(p),
                    "content": text[:100000],
                    "size": len(text),
                    "output": f"Read PDF: {p.name} ({len(text)} chars)"
                }
        except ImportError:
            pass
        except Exception:
            pass

        # Method 3: Try reading raw bytes and extracting what we can
        try:
            raw = p.read_bytes()
            # Extract readable strings from PDF
            import re as _re
            strings = _re.findall(rb'[\x20-\x7e]{20,}', raw)
            text = "\n".join(s.decode("ascii", errors="ignore") for s in strings[:500])
            if text.strip():
                return {
                    "ok": True,
                    "path": str(p),
                    "content": text[:100000],
                    "size": len(text),
                    "output": f"Read PDF (raw extraction): {p.name} ({len(text)} chars)",
                    "warning": "Raw extraction - install pypdf for better results: pip install pypdf"
                }
        except Exception:
            pass

        return {
            "ok": False,
            "error": f"Cannot read PDF: {p.name}. Install pypdf: pip install pypdf",
            "path": str(p)
        }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file"""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.base_dir / p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "path": str(p),
                "size": len(content),
                "output": f"Wrote {len(content)} chars to {p.name}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def list_files(self, path: str = None) -> Dict[str, Any]:
        """List files and folders in a directory"""
        try:
            p = Path(path) if path else self.base_dir
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try common variations before giving up
                candidates = [
                    self.base_dir / "AUREON_FOUNDATION" / p.name,
                    self.base_dir / "ALL_REPOS" / p.name,
                    self.base_dir / "AUREON_FOUNDATION",
                ]
                found = False
                for c in candidates:
                    if c.exists():
                        p = c
                        found = True
                        break
                
                if not found:
                    # PowerShell fallback: find directory by name
                    try:
                        safe_name = p.name.replace("'", "''")
                        ps_cmd = f'Get-ChildItem -Path "C:\\AUREON_AUTONOMOUS" -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object {{ $_.Name -like "*{safe_name}*" }} | Select-Object -First 1 -ExpandProperty FullName'
                        ps_result = subprocess.run(
                            ["powershell", "-Command", ps_cmd],
                            capture_output=True, text=True, timeout=10,
                            cwd=str(self.base_dir)
                        )
                        if ps_result.returncode == 0 and ps_result.stdout.strip():
                            found_path = ps_result.stdout.strip().split('\n')[0].strip()
                            if Path(found_path).exists():
                                p = Path(found_path)
                                found = True
                    except Exception:
                        pass
                
                if not found:
                    return {"ok": False, "error": f"Path not found: {path}"}
            
            if p.is_file():
                return {"ok": True, "type": "file", "path": str(p), "size": p.stat().st_size}

            items = []
            skip_dirs = {"__pycache__", ".git", "node_modules", "BROWSER_PROFILE", "driver"}
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.name in skip_dirs:
                    continue
                try:
                    items.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })
                except Exception:
                    continue

            # Build a clear output string showing dirs vs files
            dir_count = sum(1 for i in items if i["type"] == "dir")
            file_count = sum(1 for i in items if i["type"] == "file")
            output_lines = []
            for i in items:
                if i["type"] == "dir":
                    output_lines.append(f"  [FOLDER] [DIR] {i['name']}/")
                else:
                    size = i.get("size", 0) or 0
                    output_lines.append(f"  [FILE] {i['name']} ({size} bytes)")
            
            output_text = f"Listed {len(items)} items in {p.name or str(p)} ({dir_count} directories, {file_count} files):\n" + "\n".join(output_lines)
            
            return {
                "ok": True,
                "path": str(p),
                "items": items,
                "count": len(items),
                "output": output_text
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    
    def read_directory(self, path: str, pattern: str = "*") -> Dict[str, Any]:
        """Read ALL text files in a directory at once. Default: all files.
        This is much faster than listing + reading one by one.
        Skips binary files, .py files, .zip files, and images."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try AUREON_FOUNDATION subdirectory
                alt = self.base_dir / "AUREON_FOUNDATION" / p.name
                if alt.exists():
                    p = alt
                else:
                    return {"ok": False, "error": f"Directory not found: {path}"}
            if not p.is_dir():
                return {"ok": False, "error": f"Not a directory: {path}"}
            
            # Collect all text-like files (including in immediate subdirectories)
            skip_extensions = {'.py', '.pyc', '.zip', '.exe', '.dll', '.png', '.jpg', '.gif', 
                             '.ico', '.woff', '.woff2', '.ttf', '.eot', '.svg', '.mp3', '.mp4',
                             '.wav', '.pdf', '.pptx', '.xlsx', '.docx', '.db', '.sqlite'}
            skip_dirs = {"__pycache__", ".git", "node_modules", "BROWSER_PROFILE", "driver"}
            # Poisoned chatbot-era files — never return these
            skip_filenames = {
                "aureon_identity_kernel.md", "aureon_behaviour_matrix.md",
                "aureon_compiled_identity.md", "aureon_companion_system_prompt.md",
                "aureon_standard_system_prompt.md", "aureon_system_prompts.md",
                "aureon_interaction_protocol.md", "aureon_cooperative_modes.md",
                "aureon_top500_crucial_files.md", "aureon_master_system_prompt.md",
            }
            
            files = []
            for item in sorted(p.iterdir()):
                if item.name in skip_dirs:
                    continue
                if item.is_file() and item.suffix.lower() not in skip_extensions:
                    if item.name.lower() not in skip_filenames:
                        files.append(item)
                elif item.is_dir() and item.name not in skip_dirs:
                    # Read one level of subdirectories too
                    for subitem in sorted(item.iterdir()):
                        if subitem.is_file() and subitem.suffix.lower() not in skip_extensions:
                            if subitem.name.lower() not in skip_filenames:
                                files.append(subitem)
            
            results = {}
            total_chars = 0
            max_per_file = 6000  # Cap per file
            max_total = 60000   # Total cap (increased from 50K)
            
            for f in files:
                if total_chars >= max_total:
                    break
                try:
                    # Skip files > 200KB (likely not text)
                    if f.stat().st_size > 200000:
                        continue
                    content = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if content:
                        trimmed = content[:max_per_file]
                        # Use relative path from the target dir for the key
                        rel_key = str(f.relative_to(p)) if f.is_relative_to(p) else f.name
                        results[rel_key] = trimmed
                        total_chars += len(trimmed)
                except Exception:
                    continue
            
            # Build combined content string
            combined = ""
            for name, content in results.items():
                combined += f"\n{'='*60}\nFILE: {name}\n{'='*60}\n{content}\n"
            
            return {
                "ok": True,
                "path": str(p),
                "files_read": len(results),
                "total_files": len(files),
                "total_chars": total_chars,
                "content": combined,
                "file_names": list(results.keys()),
                "output": f"Read {len(results)} files ({total_chars} chars) from {p.name}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def run_command(self, command: str) -> Dict[str, Any]:
        """Run a shell command (PowerShell on Windows)"""
        if not command.strip():
            return {"ok": False, "error": "empty_command"}
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.base_dir)
            )
            output = (result.stdout or "") + (result.stderr or "")
            return {
                "ok": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:10000],
                "stderr": result.stderr[:5000],
                "output": output.strip()[:10000] or "(no output)"
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "command_timed_out_30s"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # ??????????????????????????????????????????????????????????
    # COMPOUND OPERATIONS (high-level shortcuts)
    # ??????????????????????????????????????????????????????????

    def google_search(self, query: str) -> Dict[str, Any]:
        """Search Google AND read the results page. Returns search results as text.
        This is the CORRECT way to search — not go_to_url + type + press separately."""
        if not self._ensure_browser():
            return {"ok": False, "error": "browser_not_connected"}
        try:
            import urllib.parse
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            self.driver.get(url)
            time.sleep(2)  # Wait for results to load
            
            # Read the results
            text = self.driver.find_element(By.TAG_NAME, "body").text
            total_len = len(text)
            # Keep first 6000 chars of search results (the actual result snippets)
            if total_len > 6000:
                text = text[:6000] + "\n...[truncated]..."
            
            # Also collect clickable links
            links = []
            try:
                result_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.g a, div.tF2Cxc a")
                for elem in result_elements[:10]:
                    href = elem.get_attribute("href")
                    title = elem.text.strip()
                    if href and title and "google.com" not in href:
                        links.append({"title": title[:80], "url": href})
            except Exception:
                pass
            
            return {
                "ok": True,
                "text": text,
                "links": links,
                "query": query,
                "length": total_len,
                "output": f"Google search '{query}': {total_len} chars, {len(links)} links found"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def open_link(self, url: str) -> Dict[str, Any]:
        """Open a URL and read the page content. Returns page text."""
        if not self._ensure_browser():
            return {"ok": False, "error": "browser_not_connected"}
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page load
            text = self.driver.find_element(By.TAG_NAME, "body").text
            total_len = len(text)
            if total_len > 8000:
                text = text[:8000] + "\n...[truncated]..."
            return {
                "ok": True,
                "text": text,
                "url": url,
                "title": self.driver.title,
                "length": total_len,
                "output": f"Read {total_len} chars from {self.driver.title[:60]}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # ??????????????????????????????????????????????????????????
    # BROWSER OPERATIONS (need Selenium connection)
    # ??????????????????????????????????????????????????????????

    def click_on_text(self, text: str) -> Dict[str, Any]:
        """Click any element containing this text"""
        try:
            strategies = [
                f"//button[contains(., '{text}')]",
                f"//a[contains(., '{text}')]",
                f"//span[contains(., '{text}')]",
                f"//div[contains(., '{text}') and (@role='button' or @role='tab' or @role='link')]",
                f"//input[@value='{text}']",
                f"//*[text()='{text}']",
                f"//*[contains(text(), '{text}')]",
            ]

            for xpath in strategies:
                try:
                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    element.click()
                    return {"ok": True, "output": f"Clicked '{text}'"}
                except Exception:
                    continue

            return {"ok": False, "error": f"Could not find clickable element with text '{text}'"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def find_and_click(self, description: str) -> Dict[str, Any]:
        """Find and click element by description (aria-label, title, placeholder, etc)"""
        try:
            strategies = [
                f"//*[@aria-label='{description}']",
                f"//*[@title='{description}']",
                f"//*[@placeholder='{description}']",
                f"//*[contains(@aria-label, '{description}')]",
                f"//*[contains(@title, '{description}')]",
                f"//button[contains(., '{description}')]",
                f"//a[contains(., '{description}')]",
            ]

            for xpath in strategies:
                try:
                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    element.click()
                    return {"ok": True, "output": f"Clicked element matching '{description}'"}
                except Exception:
                    continue

            return {"ok": False, "error": f"Could not find element matching '{description}'"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def type_text(self, text: str, clear_first: bool = False) -> Dict[str, Any]:
        """Type text into the currently focused element"""
        try:
            active = self.driver.switch_to.active_element
            if clear_first:
                active.clear()
            active.send_keys(text)
            return {"ok": True, "output": f"Typed {len(text)} characters"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def type_into(self, selector: str, text: str, by: str = "css") -> Dict[str, Any]:
        """Type text into a specific element found by CSS or XPath"""
        try:
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            element = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((by_method, selector))
            )
            element.click()
            time.sleep(0.2)
            element.clear()
            element.send_keys(text)
            return {"ok": True, "output": f"Typed '{text[:50]}' into {selector}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a key"""
        try:
            key_map = {
                "enter": Keys.ENTER, "return": Keys.ENTER,
                "tab": Keys.TAB,
                "escape": Keys.ESCAPE, "esc": Keys.ESCAPE,
                "backspace": Keys.BACKSPACE,
                "delete": Keys.DELETE,
                "space": Keys.SPACE,
                "up": Keys.UP, "down": Keys.DOWN,
                "left": Keys.LEFT, "right": Keys.RIGHT,
                "home": Keys.HOME, "end": Keys.END,
                "pageup": Keys.PAGE_UP, "pagedown": Keys.PAGE_DOWN,
                "f5": Keys.F5, "f11": Keys.F11, "f12": Keys.F12,
            }
            key_obj = key_map.get(key.lower(), key)
            active = self.driver.switch_to.active_element
            active.send_keys(key_obj)
            return {"ok": True, "output": f"Pressed {key}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def hotkey(self, keys: List[str]) -> Dict[str, Any]:
        """Press a key combination like Ctrl+T, Ctrl+L, etc"""
        try:
            key_map = {
                "ctrl": Keys.CONTROL, "control": Keys.CONTROL,
                "alt": Keys.ALT,
                "shift": Keys.SHIFT,
                "enter": Keys.ENTER, "return": Keys.ENTER,
                "tab": Keys.TAB,
                "escape": Keys.ESCAPE, "esc": Keys.ESCAPE,
                "backspace": Keys.BACKSPACE,
                "delete": Keys.DELETE,
                "space": Keys.SPACE,
                "f5": Keys.F5,
            }

            chain = ActionChains(self.driver)
            # Hold modifier keys, press the last key, release
            modifiers = keys[:-1]
            final_key = keys[-1]

            for mod in modifiers:
                chain.key_down(key_map.get(mod.lower(), mod))
            chain.send_keys(key_map.get(final_key.lower(), final_key))
            for mod in reversed(modifiers):
                chain.key_up(key_map.get(mod.lower(), mod))

            chain.perform()
            return {"ok": True, "output": f"Pressed {'+'.join(keys)}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def click_active(self) -> Dict[str, Any]:
        """Click the currently active/focused element"""
        try:
            active = self.driver.switch_to.active_element
            active.click()
            return {"ok": True, "output": "Clicked active element"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def go_to_url(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL — ALWAYS in a new tab to protect the chat interface."""
        try:
            if not url.startswith("http"):
                url = "https://" + url
            
            # Check if current tab is the Aureon chat interface — NEVER overwrite it
            current_url = self.driver.current_url
            is_chat_tab = "127.0.0.1" in current_url or "localhost" in current_url
            
            if is_chat_tab:
                # Open in new tab to protect chat interface
                self.driver.execute_script("window.open('');")
                self.driver.switch_to.window(self.driver.window_handles[-1])
            
            self.driver.get(url)
            time.sleep(2)
            return {"ok": True, "url": url, "title": self.driver.title, "output": f"Navigated to {url}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def return_to_chat(self) -> Dict[str, Any]:
        """Switch back to the Aureon chat tab."""
        try:
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                if "127.0.0.1" in self.driver.current_url or "localhost" in self.driver.current_url or "aureon" in self.driver.title.lower():
                    return {"ok": True, "output": "Returned to Aureon chat tab"}
            return {"ok": False, "error": "Chat tab not found"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def switch_to_tab(self, title: str) -> Dict[str, Any]:
        """Switch to a tab containing this title or URL.
        NEVER switches to the Aureon chat tab unless explicitly asked for 'aureon' or 'chat'."""
        try:
            original = self.driver.current_window_handle
            title_lower = title.lower().strip()
            
            # Collect all tabs info first
            tab_info = []
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                tab_title = self.driver.title
                tab_url = self.driver.current_url
                is_aureon_chat = ("127.0.0.1" in tab_url or "localhost" in tab_url or 
                                  "aureon" in tab_title.lower())
                tab_info.append({
                    "handle": handle,
                    "title": tab_title,
                    "url": tab_url,
                    "is_aureon_chat": is_aureon_chat,
                })
            
            # Switch back to original while we search
            self.driver.switch_to.window(original)
            
            # Determine if user wants the Aureon chat tab
            wants_aureon = any(w in title_lower for w in ['aureon', 'chat', '127.0.0.1', 'localhost'])
            
            # Search for matching tab
            for tab in tab_info:
                # Skip the Aureon chat tab UNLESS they specifically asked for it
                if tab["is_aureon_chat"] and not wants_aureon:
                    continue
                
                # Match by title or URL
                if (title_lower in tab["title"].lower() or 
                    title_lower in tab["url"].lower()):
                    self.driver.switch_to.window(tab["handle"])
                    return {"ok": True, "title": tab["title"], "url": tab["url"],
                            "output": f"Switched to '{tab['title']}'"}
            
            # Not found — list available tabs
            available = [f"'{t['title']}' ({t['url'][:50]})" for t in tab_info]
            self.driver.switch_to.window(original)
            return {"ok": False, "error": f"No tab matching '{title}'. Available: {available}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Open a new tab"""
        try:
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            if url:
                return self.go_to_url(url)
            return {"ok": True, "output": "Opened new tab"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        """Scroll the page"""
        try:
            scroll_amount = amount if direction == "down" else -amount
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            return {"ok": True, "output": f"Scrolled {direction} {amount}px"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_page_text(self) -> Dict[str, Any]:
        """Get visible text on the current page. Smart truncation: keeps last portion (latest messages)."""
        try:
            text = self.driver.find_element(By.TAG_NAME, "body").text
            total_len = len(text)
            # For chat pages (long conversations), keep the LAST 8000 chars
            # which contains the most recent messages — not the beginning
            if total_len > 8000:
                text = "...[earlier content truncated]...\n" + text[-8000:]
            return {"ok": True, "text": text, "length": total_len, "output": f"Read {total_len} chars from page"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_page_title(self) -> Dict[str, Any]:
        """Get current page title and URL"""
        try:
            return {
                "ok": True,
                "title": self.driver.title,
                "url": self.driver.current_url,
                "output": f"Page: {self.driver.title} ({self.driver.current_url})"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_tabs(self) -> Dict[str, Any]:
        """List all open browser tabs"""
        try:
            current = self.driver.current_window_handle
            tabs = []
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                tabs.append({
                    "title": self.driver.title,
                    "url": self.driver.current_url,
                    "active": handle == current,
                })
            self.driver.switch_to.window(current)
            return {"ok": True, "tabs": tabs, "count": len(tabs), "output": f"Found {len(tabs)} tabs"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def find_element(self, selector: str, by: str = "css") -> Dict[str, Any]:
        """Find element and return info about it"""
        try:
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            el = self.driver.find_element(by_method, selector)
            return {
                "ok": True,
                "tag": el.tag_name,
                "text": el.text[:500],
                "visible": el.is_displayed(),
                "output": f"Found <{el.tag_name}> with text '{el.text[:100]}'"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def click_element(self, selector: str, by: str = "css") -> Dict[str, Any]:
        """Click element by CSS selector or XPath"""
        try:
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            el = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((by_method, selector))
            )
            el.click()
            return {"ok": True, "output": f"Clicked element {selector}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # ??????????????????????????????????????????????????????????
    # DESKTOP CONTROL — Full screen mouse, keyboard, screenshots
    # Requires: pip install pyautogui pillow
    # ??????????????????????????????????????????????????????????

    def _get_pyautogui(self):
        """Lazy import pyautogui."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.3
            return pyautogui
        except ImportError:
            return None

    def desktop_click(self, x: int, y: int) -> Dict[str, Any]:
        """Click at exact screen coordinates."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.click(x, y)
            return {"ok": True, "output": f"Clicked at ({x}, {y})"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_move(self, x: int, y: int) -> Dict[str, Any]:
        """Move mouse to screen coordinates."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.moveTo(x, y, duration=0.3)
            return {"ok": True, "output": f"Mouse moved to ({x}, {y})"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_type(self, text: str) -> Dict[str, Any]:
        """Type text using keyboard (works in ANY app, not just browser)."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.typewrite(text, interval=0.02) if text.isascii() else pag.write(text)
            return {"ok": True, "output": f"Typed {len(text)} chars"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_hotkey(self, keys: list) -> Dict[str, Any]:
        """Press keyboard shortcut (e.g., ['ctrl', 'c'] or ['alt', 'tab'])."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.hotkey(*keys)
            return {"ok": True, "output": f"Pressed {'+'.join(keys)}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_screenshot(self, region: tuple = None) -> Dict[str, Any]:
        """Take a screenshot of the full screen or a region."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            screenshot_dir = self.base_dir / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            filename = f"screen_{int(time.time())}.png"
            filepath = screenshot_dir / filename
            img = pag.screenshot(region=region)
            img.save(str(filepath))
            return {"ok": True, "path": str(filepath), "output": f"Screenshot saved: {filepath}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def taskbar_click(self, app_name: str) -> Dict[str, Any]:
        """
        Click a taskbar icon by searching for it visually.
        Uses pyautogui's image recognition OR coordinate-based clicking.
        """
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            # Get screen dimensions
            screen_w, screen_h = pag.size()
            
            # The Windows taskbar is typically at the bottom, ~40px high
            taskbar_y = screen_h - 20  # Middle of taskbar
            
            # Try to find the app by looking for its icon in the taskbar
            # Strategy: take screenshot of taskbar, look for text
            try:
                import subprocess
                # Use PowerShell to find the window and activate it
                result = subprocess.run(
                    ["powershell", "-Command",
                     f"(Get-Process | Where-Object {{$_.MainWindowTitle -like '*{app_name}*'}}).MainWindowHandle"],
                    capture_output=True, text=True, timeout=5
                )
                handles = [h.strip() for h in result.stdout.strip().split('\n') if h.strip() and h.strip() != '0']
                
                if handles:
                    # Use PowerShell to bring window to front
                    subprocess.run(
                        ["powershell", "-Command",
                         f"""
                         Add-Type @'
                         using System;
                         using System.Runtime.InteropServices;
                         public class WinAPI {{
                             [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
                             [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
                         }}
'@
                         $h = [IntPtr]{handles[0]}
                         [WinAPI]::ShowWindow($h, 9)
                         [WinAPI]::SetForegroundWindow($h)
                         """],
                        capture_output=True, timeout=5
                    )
                    return {"ok": True, "output": f"Activated window: {app_name}"}
                else:
                    return {"ok": False, "error": f"No window found matching '{app_name}'"}
            except Exception as inner:
                return {"ok": False, "error": f"Could not find {app_name}: {repr(inner)}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}
=======
import os
import subprocess
import time
from typing import Optional, List
import psutil
import pyttsx3
import speech_recognition as sr
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
"""
AUREON HANDS - COMPLETE
========================
Two capabilities that work INDEPENDENTLY:

1. FILE OPERATIONS (always work, no browser needed):
   - search_files, read_file, write_file, list_files
   - run_command (PowerShell/cmd)
   
2. BROWSER OPERATIONS (need Chrome with --remote-debugging-port=9222):
   - click_on_text, type_text, press, go_to_url, switch_tab, scroll, new_tab
   - Auto-retries connection if browser wasn't ready at startup
"""

from typing import Dict, Any, Optional, List
import subprocess
import time
import os
from pathlib import Path

try:
    from aureon_surgeon import AureonSurgeon
except ImportError:
    AureonSurgeon = None

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.action_chains import ActionChains
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class AureonHands:
    """
    AUREON's hands - controls files AND browser.
    File operations ALWAYS work.
    Browser operations auto-retry connection.
    """

    def __init__(self, base_dir: str = r"C:\AUREON_AUTONOMOUS"):
        self.base_dir = Path(base_dir)
        self.driver = None
        self.browser_connected = False
        self._last_connect_attempt = 0
        
        # Initialize the surgical code editor
        self.surgeon = AureonSurgeon(base_dir=base_dir) if AureonSurgeon else None

        if SELENIUM_AVAILABLE:
            self._try_connect_browser()
        else:
            print("\u26A0  Selenium not installed: pip install selenium")
            print("   Browser control disabled. File operations still work.")

    # ??????????????????????????????????????????????????????????
    # BROWSER CONNECTION (auto-retry)
    # ??????????????????????????????????????????????????????????

    def _try_connect_browser(self) -> bool:
        """Try to connect to existing browser. Auto-launches Chrome if needed."""
        now = time.time()
        # Don't spam retries — wait at least 10 seconds between attempts
        if now - self._last_connect_attempt < 10:
            return self.browser_connected
        self._last_connect_attempt = now

        if self.browser_connected and self.driver:
            # Check if still alive
            try:
                _ = self.driver.title
                return True
            except Exception:
                self.browser_connected = False
                self.driver = None

        # Step 1: Check if Chrome is already listening on 9222
        import socket
        chrome_listening = False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(("127.0.0.1", 9222))
            s.close()
            chrome_listening = True
        except Exception:
            pass
        
        # Step 2: If Chrome is NOT listening, launch it ourselves
        if not chrome_listening:
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ]
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ]
            
            launched = False
            for browser_path in chrome_paths + edge_paths:
                if os.path.exists(browser_path):
                    try:
                        browser_name = "Chrome" if "chrome" in browser_path.lower() else "Edge"
                        print(f"   [LAUNCH] Launching {browser_name} with --remote-debugging-port=9222...")
                        subprocess.Popen(
                            [browser_path, "--remote-debugging-port=9222", "--no-first-run"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        # Wait for it to start
                        for attempt in range(15):  # Up to 15 seconds
                            time.sleep(1)
                            try:
                                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                s.settimeout(1)
                                s.connect(("127.0.0.1", 9222))
                                s.close()
                                print(f"   [OK] {browser_name} is now listening on port 9222")
                                launched = True
                                break
                            except Exception:
                                pass
                        if launched:
                            break
                    except Exception as launch_err:
                        print(f"   [WARN] Failed to launch {browser_path}: {launch_err}")
            
            if not launched:
                print("   [FAIL] Could not launch any browser with debug port")
                return False

        # Step 3: Connect via Selenium
        # Try Chrome
        try:
            options = webdriver.ChromeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Chrome(options=options)
            self.browser_connected = True
            print("\u2705 Connected to Chrome (port 9222)")
            return True
        except Exception as chrome_err:
            short_err = str(chrome_err).split('\n')[0][:120]
            print(f"   Chrome selenium failed: {short_err}")

        # Try Edge
        try:
            options = webdriver.EdgeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Edge(options=options)
            self.browser_connected = True
            print("\u2705 Connected to Edge (port 9222)")
            return True
        except Exception as edge_err:
            short_err = str(edge_err).split('\n')[0][:120]
            print(f"   Edge selenium failed: {short_err}")

        if not self.browser_connected:
            print("\u26A0  Browser launched but Selenium could not connect")
            print("    This usually means chromedriver version doesn't match Chrome version")
            print("    Update: pip install --upgrade selenium")
        return False

    def _ensure_browser(self) -> bool:
        """Ensure browser is connected, retry if needed."""
        if self.browser_connected:
            try:
                _ = self.driver.title
                return True
            except Exception:
                self.browser_connected = False
                self.driver = None

        if SELENIUM_AVAILABLE:
            return self._try_connect_browser()
        return False

    # ??????????????????????????????????????????????????????????
    # DISPATCH - routes all operations
    # ??????????????????????????????????????????????????????????

    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """Route operations to handlers"""

        # ?? File operations (ALWAYS work) ????????????
        if op == "search_files":
            return self.search_files(kwargs.get("query", ""), kwargs.get("root", str(self.base_dir)))
        elif op == "read_file":
            return self.read_file(kwargs.get("path", ""))
        elif op == "write_file":
            return self.write_file(kwargs.get("path", ""), kwargs.get("content", ""))
        elif op == "list_files":
            return self.list_files(kwargs.get("path", str(self.base_dir)))
        elif op == "read_directory":
            return self.read_directory(kwargs.get("path", ""), kwargs.get("pattern", "*.md"))
        elif op == "scan_all_files":
            return self.scan_all_files(
                root=kwargs.get("root", kwargs.get("path", str(self.base_dir))),
                extensions=kwargs.get("extensions", ".md,.py,.kernel")
            )
        elif op == "run_command":
            return self.run_command(kwargs.get("command", ""))
        elif op == "google_search":
            return self.google_search(kwargs.get("query", ""))
        elif op == "open_link":
            return self.open_link(kwargs.get("url", ""))
        
        # ?? Surgical code operations ?????
        elif op in ("scan_file", "select_focus", "apply_edit", "verify_syntax", "revert", "mutation_log"):
            if self.surgeon:
                return self.surgeon.dispatch(op, **kwargs)
            else:
                return {"ok": False, "error": "Surgeon not available — install aureon_surgeon.py"}

        # ?? Browser operations (need connection) ?????
        if not self._ensure_browser():
            return {
                "ok": False,
                "error": "browser_not_connected",
                "hint": "Start Chrome with: Start-Process 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe' -ArgumentList '--remote-debugging-port=9222'",
            }

        if op == "click_on_text":
            return self.click_on_text(kwargs.get("text", ""))
        elif op == "find_and_click":
            return self.find_and_click(kwargs.get("description", ""))
        elif op == "type_text":
            return self.type_text(kwargs.get("text", ""), kwargs.get("clear_first", False))
        elif op == "press":
            return self.press_key(kwargs.get("key", "enter"))
        elif op == "hotkey":
            return self.hotkey(kwargs.get("keys", []))
        elif op == "click":
            return self.click_active()
        elif op == "go_to_url":
            return self.go_to_url(kwargs.get("url", ""))
        elif op == "switch_tab":
            return self.switch_to_tab(kwargs.get("title", kwargs.get("title_contains", "")))
        elif op == "focus_window":
            return self.switch_to_tab(kwargs.get("title_contains", ""))
        elif op == "scroll":
            amt = kwargs.get("amount", 300)
            direction = kwargs.get("direction", "down" if amt >= 0 else "up")
            return self.scroll(direction, abs(amt))
        elif op == "new_tab":
            return self.new_tab(kwargs.get("url"))
        elif op == "get_page_text":
            return self.get_page_text()
        elif op == "get_page_title":
            return self.get_page_title()
        elif op == "get_tabs":
            return self.get_tabs()
        elif op == "return_to_chat":
            return self.return_to_chat()
        elif op == "find_element":
            return self.find_element(kwargs.get("css", kwargs.get("xpath", "")), kwargs.get("by", "css"))
        elif op == "click_element":
            return self.click_element(kwargs.get("css", kwargs.get("xpath", "")), kwargs.get("by", "css"))
        elif op == "type_into":
            return self.type_into(kwargs.get("css", kwargs.get("xpath", "")), kwargs.get("text", ""), kwargs.get("by", "css"))
        else:
            # ?? Desktop operations (pyautogui — controls ENTIRE screen) ??
            if op == "desktop_click":
                return self.desktop_click(kwargs.get("x", 0), kwargs.get("y", 0))
            elif op == "desktop_move":
                return self.desktop_move(kwargs.get("x", 0), kwargs.get("y", 0))
            elif op == "desktop_type":
                return self.desktop_type(kwargs.get("text", ""))
            elif op == "desktop_hotkey":
                return self.desktop_hotkey(kwargs.get("keys", []))
            elif op == "desktop_screenshot":
                return self.desktop_screenshot()
            elif op == "taskbar_click":
                return self.taskbar_click(kwargs.get("app_name", ""))
            else:
                return {"ok": False, "error": f"unknown_operation: {op}"}

    # ??????????????????????????????????????????????????????????
    def scan_all_files(self, root: str = None, extensions: str = ".md,.py,.kernel") -> Dict[str, Any]:
        """
        Scan a directory and return ALL files matching given extensions.
        Unlike search_files, this doesn't search content - just lists by type.
        
        Args:
            root: Directory to scan (defaults to base_dir)
            extensions: Comma-separated extensions like ".md,.py,.kernel"
        """
        try:
            p = Path(root) if root else self.base_dir
            if not p.exists():
                return {"ok": False, "error": f"Directory not found: {root}"}
            
            ext_list = [e.strip().lower() for e in extensions.split(",")]
            # Ensure dots
            ext_list = [e if e.startswith(".") else f".{e}" for e in ext_list]
            
            skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", 
                        "BROWSER_PROFILE", "driver", "msedgedriver"}
            
            files = []
            for f in p.rglob("*"):
                if f.is_dir():
                    continue
                if any(sd in f.parts for sd in skip_dirs):
                    continue
                if f.suffix.lower() in ext_list:
                    files.append({
                        "path": str(f),
                        "name": f.name,
                        "type": f.suffix.lower(),
                        "size": f.stat().st_size,
                        "relative": str(f.relative_to(p)) if f.is_relative_to(p) else f.name,
                    })
            
            # Sort by type then name
            files.sort(key=lambda x: (x["type"], x["name"].lower()))
            
            # Build summary by type
            type_counts = {}
            for f in files:
                t = f["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
            
            summary_parts = [f"{count} {ext} files" for ext, count in sorted(type_counts.items())]
            summary = ", ".join(summary_parts)
            
            return {
                "ok": True,
                "files": files,
                "count": len(files),
                "by_type": type_counts,
                "output": f"Found {len(files)} files ({summary}) in {p.name}",
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # FILE OPERATIONS (always work, no browser needed)
    # ??????????????????????????????????????????????????????????

    def search_files(self, query: str, root: str = None) -> Dict[str, Any]:
        """Search for files containing text or matching name.
        Prioritizes filename matches over content matches.
        Skips code files (.py) and huge files (>200KB) for content search."""
        try:
            p = Path(root) if root else self.base_dir
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try common variations before giving up
                for candidate in [
                    self.base_dir / "AUREON_FOUNDATION",
                    self.base_dir,
                ]:
                    if candidate.exists():
                        p = candidate
                        break
            filename_matches = []
            content_matches = []
            extensions = (".txt", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg", ".html", ".css", ".js", ".csv", ".log")
            skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", "BROWSER_PROFILE", "driver"}
            # Files that should never be returned as content matches
            skip_content_files = {
                "aureon_brain.py", "aureon_web_interface.py", "aureon_hands.py",
                "aureon_kernel_loader.py", "AUREON_COMPILED_IDENTITY.md",
            }

            for f in p.rglob("*"):
                if len(filename_matches) + len(content_matches) >= 30:
                    break
                if f.is_dir():
                    continue
                if any(sd in f.parts for sd in skip_dirs):
                    continue

                # Match filename first (always include, even .py)
                if query.lower() in f.name.lower():
                    filename_matches.append({"path": str(f), "match": "filename"})
                    continue

                # Skip code files and huge files for content search
                if f.name in skip_content_files:
                    continue
                if f.suffix.lower() == ".py":
                    continue
                if f.stat().st_size > 200000:  # Skip files > 200KB
                    continue

                # Match content for text files
                if f.suffix.lower() in extensions:
                    try:
                        text = f.read_text(encoding="utf-8", errors="ignore")[:50000]
                        if query.lower() in text.lower():
                            for line in text.split("\n"):
                                if query.lower() in line.lower():
                                    content_matches.append({
                                        "path": str(f),
                                        "match": "content",
                                        "context": line.strip()[:200]
                                    })
                                    break
                    except Exception:
                        continue

            # Filename matches first, then content matches
            matches = filename_matches + content_matches
            return {
                "ok": True,
                "matches": matches,
                "count": len(matches),
                "output": f"Found {len(matches)} files matching '{query}' ({len(filename_matches)} by name, {len(content_matches)} by content)"
            }
        except Exception as e:
            # POWERSHELL FALLBACK: If Python path traversal fails, use PowerShell
            try:
                safe_query = query.replace("'", "''").replace('"', '""')
                # Search by filename first
                ps_cmd = (
                    f'Get-ChildItem -Path "C:\\AUREON_AUTONOMOUS" -Recurse -ErrorAction SilentlyContinue | '
                    f'Where-Object {{ $_.Name -like "*{safe_query.replace(" ", "*")}*" }} | '
                    f'Select-Object -First 15 FullName'
                )
                ps_result = subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    capture_output=True, text=True, timeout=15,
                    cwd=str(self.base_dir)
                )
                if ps_result.returncode == 0 and ps_result.stdout.strip():
                    lines = [l.strip() for l in ps_result.stdout.strip().split('\n') if l.strip() and not l.strip().startswith('-')]
                    # Skip header line
                    paths = [l for l in lines if '\\' in l or '/' in l]
                    if paths:
                        matches = [{"path": p, "match": "powershell_filename"} for p in paths]
                        return {
                            "ok": True,
                            "matches": matches,
                            "count": len(matches),
                            "output": f"PowerShell found {len(matches)} files matching '{query}'"
                        }
            except Exception:
                pass
            return {"ok": False, "error": repr(e)}

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file's contents - supports text files AND PDFs"""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try searching for it by name
                for match in self.base_dir.rglob(f"*{Path(path).name}*"):
                    if match.is_file():
                        p = match
                        break
                else:
                    return {"ok": False, "error": f"File not found: {path}"}

            # Handle PDFs
            if p.suffix.lower() == ".pdf":
                return self._read_pdf(p)

            content = p.read_text(encoding="utf-8", errors="ignore")
            return {
                "ok": True,
                "path": str(p),
                "content": content[:100000],
                "size": len(content),
                "output": f"Read {len(content)} chars from {p.name}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def _read_pdf(self, p: Path) -> Dict[str, Any]:
        """Extract text from a PDF file"""
        # Try multiple PDF libraries
        text = ""

        # Method 1: PyPDF2 / pypdf
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader

            reader = PdfReader(str(p))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"--- Page {i+1} ---\n{page_text}")
            text = "\n\n".join(pages)
            if text.strip():
                return {
                    "ok": True,
                    "path": str(p),
                    "content": text[:100000],
                    "pages": len(reader.pages),
                    "size": len(text),
                    "output": f"Read PDF: {p.name} ({len(reader.pages)} pages, {len(text)} chars)"
                }
        except ImportError:
            pass
        except Exception:
            pass

        # Method 2: pdfminer
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(str(p))
            if text.strip():
                return {
                    "ok": True,
                    "path": str(p),
                    "content": text[:100000],
                    "size": len(text),
                    "output": f"Read PDF: {p.name} ({len(text)} chars)"
                }
        except ImportError:
            pass
        except Exception:
            pass

        # Method 3: Try reading raw bytes and extracting what we can
        try:
            raw = p.read_bytes()
            # Extract readable strings from PDF
            import re as _re
            strings = _re.findall(rb'[\x20-\x7e]{20,}', raw)
            text = "\n".join(s.decode("ascii", errors="ignore") for s in strings[:500])
            if text.strip():
                return {
                    "ok": True,
                    "path": str(p),
                    "content": text[:100000],
                    "size": len(text),
                    "output": f"Read PDF (raw extraction): {p.name} ({len(text)} chars)",
                    "warning": "Raw extraction - install pypdf for better results: pip install pypdf"
                }
        except Exception:
            pass

        return {
            "ok": False,
            "error": f"Cannot read PDF: {p.name}. Install pypdf: pip install pypdf",
            "path": str(p)
        }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file"""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.base_dir / p
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "path": str(p),
                "size": len(content),
                "output": f"Wrote {len(content)} chars to {p.name}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def list_files(self, path: str = None) -> Dict[str, Any]:
        """List files and folders in a directory"""
        try:
            p = Path(path) if path else self.base_dir
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try common variations before giving up
                candidates = [
                    self.base_dir / "AUREON_FOUNDATION" / p.name,
                    self.base_dir / "ALL_REPOS" / p.name,
                    self.base_dir / "AUREON_FOUNDATION",
                ]
                found = False
                for c in candidates:
                    if c.exists():
                        p = c
                        found = True
                        break
                
                if not found:
                    # PowerShell fallback: find directory by name
                    try:
                        safe_name = p.name.replace("'", "''")
                        ps_cmd = f'Get-ChildItem -Path "C:\\AUREON_AUTONOMOUS" -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object {{ $_.Name -like "*{safe_name}*" }} | Select-Object -First 1 -ExpandProperty FullName'
                        ps_result = subprocess.run(
                            ["powershell", "-Command", ps_cmd],
                            capture_output=True, text=True, timeout=10,
                            cwd=str(self.base_dir)
                        )
                        if ps_result.returncode == 0 and ps_result.stdout.strip():
                            found_path = ps_result.stdout.strip().split('\n')[0].strip()
                            if Path(found_path).exists():
                                p = Path(found_path)
                                found = True
                    except Exception:
                        pass
                
                if not found:
                    return {"ok": False, "error": f"Path not found: {path}"}
            
            if p.is_file():
                return {"ok": True, "type": "file", "path": str(p), "size": p.stat().st_size}

            items = []
            skip_dirs = {"__pycache__", ".git", "node_modules", "BROWSER_PROFILE", "driver"}
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.name in skip_dirs:
                    continue
                try:
                    items.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })
                except Exception:
                    continue

            # Build a clear output string showing dirs vs files
            dir_count = sum(1 for i in items if i["type"] == "dir")
            file_count = sum(1 for i in items if i["type"] == "file")
            output_lines = []
            for i in items:
                if i["type"] == "dir":
                    output_lines.append(f"  [FOLDER] [DIR] {i['name']}/")
                else:
                    size = i.get("size", 0) or 0
                    output_lines.append(f"  [FILE] {i['name']} ({size} bytes)")
            
            output_text = f"Listed {len(items)} items in {p.name or str(p)} ({dir_count} directories, {file_count} files):\n" + "\n".join(output_lines)
            
            return {
                "ok": True,
                "path": str(p),
                "items": items,
                "count": len(items),
                "output": output_text
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    
    def read_directory(self, path: str, pattern: str = "*") -> Dict[str, Any]:
        """Read ALL text files in a directory at once. Default: all files.
        This is much faster than listing + reading one by one.
        Skips binary files, .py files, .zip files, and images."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = self.base_dir / p
            if not p.exists():
                # Try AUREON_FOUNDATION subdirectory
                alt = self.base_dir / "AUREON_FOUNDATION" / p.name
                if alt.exists():
                    p = alt
                else:
                    return {"ok": False, "error": f"Directory not found: {path}"}
            if not p.is_dir():
                return {"ok": False, "error": f"Not a directory: {path}"}
            
            # Collect all text-like files (including in immediate subdirectories)
            skip_extensions = {'.py', '.pyc', '.zip', '.exe', '.dll', '.png', '.jpg', '.gif', 
                             '.ico', '.woff', '.woff2', '.ttf', '.eot', '.svg', '.mp3', '.mp4',
                             '.wav', '.pdf', '.pptx', '.xlsx', '.docx', '.db', '.sqlite'}
            skip_dirs = {"__pycache__", ".git", "node_modules", "BROWSER_PROFILE", "driver"}
            # Poisoned chatbot-era files — never return these
            skip_filenames = {
                "aureon_identity_kernel.md", "aureon_behaviour_matrix.md",
                "aureon_compiled_identity.md", "aureon_companion_system_prompt.md",
                "aureon_standard_system_prompt.md", "aureon_system_prompts.md",
                "aureon_interaction_protocol.md", "aureon_cooperative_modes.md",
                "aureon_top500_crucial_files.md", "aureon_master_system_prompt.md",
            }
            
            files = []
            for item in sorted(p.iterdir()):
                if item.name in skip_dirs:
                    continue
                if item.is_file() and item.suffix.lower() not in skip_extensions:
                    if item.name.lower() not in skip_filenames:
                        files.append(item)
                elif item.is_dir() and item.name not in skip_dirs:
                    # Read one level of subdirectories too
                    for subitem in sorted(item.iterdir()):
                        if subitem.is_file() and subitem.suffix.lower() not in skip_extensions:
                            if subitem.name.lower() not in skip_filenames:
                                files.append(subitem)
            
            results = {}
            total_chars = 0
            max_per_file = 6000  # Cap per file
            max_total = 60000   # Total cap (increased from 50K)
            
            for f in files:
                if total_chars >= max_total:
                    break
                try:
                    # Skip files > 200KB (likely not text)
                    if f.stat().st_size > 200000:
                        continue
                    content = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if content:
                        trimmed = content[:max_per_file]
                        # Use relative path from the target dir for the key
                        rel_key = str(f.relative_to(p)) if f.is_relative_to(p) else f.name
                        results[rel_key] = trimmed
                        total_chars += len(trimmed)
                except Exception:
                    continue
            
            # Build combined content string
            combined = ""
            for name, content in results.items():
                combined += f"\n{'='*60}\nFILE: {name}\n{'='*60}\n{content}\n"
            
            return {
                "ok": True,
                "path": str(p),
                "files_read": len(results),
                "total_files": len(files),
                "total_chars": total_chars,
                "content": combined,
                "file_names": list(results.keys()),
                "output": f"Read {len(results)} files ({total_chars} chars) from {p.name}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def run_command(self, command: str) -> Dict[str, Any]:
        """Run a shell command (PowerShell on Windows)"""
        if not command.strip():
            return {"ok": False, "error": "empty_command"}
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True, text=True, timeout=30,
                cwd=str(self.base_dir)
            )
            output = (result.stdout or "") + (result.stderr or "")
            return {
                "ok": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[:10000],
                "stderr": result.stderr[:5000],
                "output": output.strip()[:10000] or "(no output)"
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "command_timed_out_30s"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # ??????????????????????????????????????????????????????????
    # COMPOUND OPERATIONS (high-level shortcuts)
    # ??????????????????????????????????????????????????????????

    def google_search(self, query: str) -> Dict[str, Any]:
        """Search Google AND read the results page. Returns search results as text.
        This is the CORRECT way to search — not go_to_url + type + press separately."""
        if not self._ensure_browser():
            return {"ok": False, "error": "browser_not_connected"}
        try:
            import urllib.parse
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            self.driver.get(url)
            time.sleep(2)  # Wait for results to load
            
            # Read the results
            text = self.driver.find_element(By.TAG_NAME, "body").text
            total_len = len(text)
            # Keep first 6000 chars of search results (the actual result snippets)
            if total_len > 6000:
                text = text[:6000] + "\n...[truncated]..."
            
            # Also collect clickable links
            links = []
            try:
                result_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.g a, div.tF2Cxc a")
                for elem in result_elements[:10]:
                    href = elem.get_attribute("href")
                    title = elem.text.strip()
                    if href and title and "google.com" not in href:
                        links.append({"title": title[:80], "url": href})
            except Exception:
                pass
            
            return {
                "ok": True,
                "text": text,
                "links": links,
                "query": query,
                "length": total_len,
                "output": f"Google search '{query}': {total_len} chars, {len(links)} links found"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def open_link(self, url: str) -> Dict[str, Any]:
        """Open a URL and read the page content. Returns page text."""
        if not self._ensure_browser():
            return {"ok": False, "error": "browser_not_connected"}
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page load
            text = self.driver.find_element(By.TAG_NAME, "body").text
            total_len = len(text)
            if total_len > 8000:
                text = text[:8000] + "\n...[truncated]..."
            return {
                "ok": True,
                "text": text,
                "url": url,
                "title": self.driver.title,
                "length": total_len,
                "output": f"Read {total_len} chars from {self.driver.title[:60]}"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # ??????????????????????????????????????????????????????????
    # BROWSER OPERATIONS (need Selenium connection)
    # ??????????????????????????????????????????????????????????

    def click_on_text(self, text: str) -> Dict[str, Any]:
        """Click any element containing this text"""
        try:
            strategies = [
                f"//button[contains(., '{text}')]",
                f"//a[contains(., '{text}')]",
                f"//span[contains(., '{text}')]",
                f"//div[contains(., '{text}') and (@role='button' or @role='tab' or @role='link')]",
                f"//input[@value='{text}']",
                f"//*[text()='{text}']",
                f"//*[contains(text(), '{text}')]",
            ]

            for xpath in strategies:
                try:
                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    element.click()
                    return {"ok": True, "output": f"Clicked '{text}'"}
                except Exception:
                    continue

            return {"ok": False, "error": f"Could not find clickable element with text '{text}'"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def find_and_click(self, description: str) -> Dict[str, Any]:
        """Find and click element by description (aria-label, title, placeholder, etc)"""
        try:
            strategies = [
                f"//*[@aria-label='{description}']",
                f"//*[@title='{description}']",
                f"//*[@placeholder='{description}']",
                f"//*[contains(@aria-label, '{description}')]",
                f"//*[contains(@title, '{description}')]",
                f"//button[contains(., '{description}')]",
                f"//a[contains(., '{description}')]",
            ]

            for xpath in strategies:
                try:
                    element = WebDriverWait(self.driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    element.click()
                    return {"ok": True, "output": f"Clicked element matching '{description}'"}
                except Exception:
                    continue

            return {"ok": False, "error": f"Could not find element matching '{description}'"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def type_text(self, text: str, clear_first: bool = False) -> Dict[str, Any]:
        """Type text into the currently focused element"""
        try:
            active = self.driver.switch_to.active_element
            if clear_first:
                active.clear()
            active.send_keys(text)
            return {"ok": True, "output": f"Typed {len(text)} characters"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def type_into(self, selector: str, text: str, by: str = "css") -> Dict[str, Any]:
        """Type text into a specific element found by CSS or XPath"""
        try:
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            element = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((by_method, selector))
            )
            element.click()
            time.sleep(0.2)
            element.clear()
            element.send_keys(text)
            return {"ok": True, "output": f"Typed '{text[:50]}' into {selector}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a key"""
        try:
            key_map = {
                "enter": Keys.ENTER, "return": Keys.ENTER,
                "tab": Keys.TAB,
                "escape": Keys.ESCAPE, "esc": Keys.ESCAPE,
                "backspace": Keys.BACKSPACE,
                "delete": Keys.DELETE,
                "space": Keys.SPACE,
                "up": Keys.UP, "down": Keys.DOWN,
                "left": Keys.LEFT, "right": Keys.RIGHT,
                "home": Keys.HOME, "end": Keys.END,
                "pageup": Keys.PAGE_UP, "pagedown": Keys.PAGE_DOWN,
                "f5": Keys.F5, "f11": Keys.F11, "f12": Keys.F12,
            }
            key_obj = key_map.get(key.lower(), key)
            active = self.driver.switch_to.active_element
            active.send_keys(key_obj)
            return {"ok": True, "output": f"Pressed {key}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def hotkey(self, keys: List[str]) -> Dict[str, Any]:
        """Press a key combination like Ctrl+T, Ctrl+L, etc"""
        try:
            key_map = {
                "ctrl": Keys.CONTROL, "control": Keys.CONTROL,
                "alt": Keys.ALT,
                "shift": Keys.SHIFT,
                "enter": Keys.ENTER, "return": Keys.ENTER,
                "tab": Keys.TAB,
                "escape": Keys.ESCAPE, "esc": Keys.ESCAPE,
                "backspace": Keys.BACKSPACE,
                "delete": Keys.DELETE,
                "space": Keys.SPACE,
                "f5": Keys.F5,
            }

            chain = ActionChains(self.driver)
            # Hold modifier keys, press the last key, release
            modifiers = keys[:-1]
            final_key = keys[-1]

            for mod in modifiers:
                chain.key_down(key_map.get(mod.lower(), mod))
            chain.send_keys(key_map.get(final_key.lower(), final_key))
            for mod in reversed(modifiers):
                chain.key_up(key_map.get(mod.lower(), mod))

            chain.perform()
            return {"ok": True, "output": f"Pressed {'+'.join(keys)}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def click_active(self) -> Dict[str, Any]:
        """Click the currently active/focused element"""
        try:
            active = self.driver.switch_to.active_element
            active.click()
            return {"ok": True, "output": "Clicked active element"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def go_to_url(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL — ALWAYS in a new tab to protect the chat interface."""
        try:
            if not url.startswith("http"):
                url = "https://" + url
            
            # Check if current tab is the Aureon chat interface — NEVER overwrite it
            current_url = self.driver.current_url
            is_chat_tab = "127.0.0.1" in current_url or "localhost" in current_url
            
            if is_chat_tab:
                # Open in new tab to protect chat interface
                self.driver.execute_script("window.open('');")
                self.driver.switch_to.window(self.driver.window_handles[-1])
            
            self.driver.get(url)
            time.sleep(2)
            return {"ok": True, "url": url, "title": self.driver.title, "output": f"Navigated to {url}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def return_to_chat(self) -> Dict[str, Any]:
        """Switch back to the Aureon chat tab."""
        try:
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                if "127.0.0.1" in self.driver.current_url or "localhost" in self.driver.current_url or "aureon" in self.driver.title.lower():
                    return {"ok": True, "output": "Returned to Aureon chat tab"}
            return {"ok": False, "error": "Chat tab not found"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def switch_to_tab(self, title: str) -> Dict[str, Any]:
        """Switch to a tab containing this title or URL.
        NEVER switches to the Aureon chat tab unless explicitly asked for 'aureon' or 'chat'."""
        try:
            original = self.driver.current_window_handle
            title_lower = title.lower().strip()
            
            # Collect all tabs info first
            tab_info = []
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                tab_title = self.driver.title
                tab_url = self.driver.current_url
                is_aureon_chat = ("127.0.0.1" in tab_url or "localhost" in tab_url or 
                                  "aureon" in tab_title.lower())
                tab_info.append({
                    "handle": handle,
                    "title": tab_title,
                    "url": tab_url,
                    "is_aureon_chat": is_aureon_chat,
                })
            
            # Switch back to original while we search
            self.driver.switch_to.window(original)
            
            # Determine if user wants the Aureon chat tab
            wants_aureon = any(w in title_lower for w in ['aureon', 'chat', '127.0.0.1', 'localhost'])
            
            # Search for matching tab
            for tab in tab_info:
                # Skip the Aureon chat tab UNLESS they specifically asked for it
                if tab["is_aureon_chat"] and not wants_aureon:
                    continue
                
                # Match by title or URL
                if (title_lower in tab["title"].lower() or 
                    title_lower in tab["url"].lower()):
                    self.driver.switch_to.window(tab["handle"])
                    return {"ok": True, "title": tab["title"], "url": tab["url"],
                            "output": f"Switched to '{tab['title']}'"}
            
            # Not found — list available tabs
            available = [f"'{t['title']}' ({t['url'][:50]})" for t in tab_info]
            self.driver.switch_to.window(original)
            return {"ok": False, "error": f"No tab matching '{title}'. Available: {available}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Open a new tab"""
        try:
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            if url:
                return self.go_to_url(url)
            return {"ok": True, "output": "Opened new tab"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def scroll(self, direction: str = "down", amount: int = 300) -> Dict[str, Any]:
        """Scroll the page"""
        try:
            scroll_amount = amount if direction == "down" else -amount
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            return {"ok": True, "output": f"Scrolled {direction} {amount}px"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_page_text(self) -> Dict[str, Any]:
        """Get visible text on the current page. Smart truncation: keeps last portion (latest messages)."""
        try:
            text = self.driver.find_element(By.TAG_NAME, "body").text
            total_len = len(text)
            # For chat pages (long conversations), keep the LAST 8000 chars
            # which contains the most recent messages — not the beginning
            if total_len > 8000:
                text = "...[earlier content truncated]...\n" + text[-8000:]
            return {"ok": True, "text": text, "length": total_len, "output": f"Read {total_len} chars from page"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_page_title(self) -> Dict[str, Any]:
        """Get current page title and URL"""
        try:
            return {
                "ok": True,
                "title": self.driver.title,
                "url": self.driver.current_url,
                "output": f"Page: {self.driver.title} ({self.driver.current_url})"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def get_tabs(self) -> Dict[str, Any]:
        """List all open browser tabs"""
        try:
            current = self.driver.current_window_handle
            tabs = []
            for handle in self.driver.window_handles:
                self.driver.switch_to.window(handle)
                tabs.append({
                    "title": self.driver.title,
                    "url": self.driver.current_url,
                    "active": handle == current,
                })
            self.driver.switch_to.window(current)
            return {"ok": True, "tabs": tabs, "count": len(tabs), "output": f"Found {len(tabs)} tabs"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def find_element(self, selector: str, by: str = "css") -> Dict[str, Any]:
        """Find element and return info about it"""
        try:
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            el = self.driver.find_element(by_method, selector)
            return {
                "ok": True,
                "tag": el.tag_name,
                "text": el.text[:500],
                "visible": el.is_displayed(),
                "output": f"Found <{el.tag_name}> with text '{el.text[:100]}'"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def click_element(self, selector: str, by: str = "css") -> Dict[str, Any]:
        """Click element by CSS selector or XPath"""
        try:
            by_method = By.CSS_SELECTOR if by == "css" else By.XPATH
            el = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((by_method, selector))
            )
            el.click()
            return {"ok": True, "output": f"Clicked element {selector}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    # ??????????????????????????????????????????????????????????
    # DESKTOP CONTROL — Full screen mouse, keyboard, screenshots
    # Requires: pip install pyautogui pillow
    # ??????????????????????????????????????????????????????????

    def _get_pyautogui(self):
        """Lazy import pyautogui."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.3
            return pyautogui
        except ImportError:
            return None

    def desktop_click(self, x: int, y: int) -> Dict[str, Any]:
        """Click at exact screen coordinates."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.click(x, y)
            return {"ok": True, "output": f"Clicked at ({x}, {y})"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_move(self, x: int, y: int) -> Dict[str, Any]:
        """Move mouse to screen coordinates."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.moveTo(x, y, duration=0.3)
            return {"ok": True, "output": f"Mouse moved to ({x}, {y})"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_type(self, text: str) -> Dict[str, Any]:
        """Type text using keyboard (works in ANY app, not just browser)."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.typewrite(text, interval=0.02) if text.isascii() else pag.write(text)
            return {"ok": True, "output": f"Typed {len(text)} chars"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_hotkey(self, keys: list) -> Dict[str, Any]:
        """Press keyboard shortcut (e.g., ['ctrl', 'c'] or ['alt', 'tab'])."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            pag.hotkey(*keys)
            return {"ok": True, "output": f"Pressed {'+'.join(keys)}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def desktop_screenshot(self, region: tuple = None) -> Dict[str, Any]:
        """Take a screenshot of the full screen or a region."""
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            screenshot_dir = self.base_dir / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            filename = f"screen_{int(time.time())}.png"
            filepath = screenshot_dir / filename
            img = pag.screenshot(region=region)
            img.save(str(filepath))
            return {"ok": True, "path": str(filepath), "output": f"Screenshot saved: {filepath}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}

    def taskbar_click(self, app_name: str) -> Dict[str, Any]:
        """
        Click a taskbar icon by searching for it visually.
        Uses pyautogui's image recognition OR coordinate-based clicking.
        """
        pag = self._get_pyautogui()
        if not pag:
            return {"ok": False, "error": "pyautogui not installed. Run: pip install pyautogui"}
        try:
            # Get screen dimensions
            screen_w, screen_h = pag.size()
            
            # The Windows taskbar is typically at the bottom, ~40px high
            taskbar_y = screen_h - 20  # Middle of taskbar
            
            # Try to find the app by looking for its icon in the taskbar
            # Strategy: take screenshot of taskbar, look for text
            try:
                import subprocess
                # Use PowerShell to find the window and activate it
                result = subprocess.run(
                    ["powershell", "-Command",
                     f"(Get-Process | Where-Object {{$_.MainWindowTitle -like '*{app_name}*'}}).MainWindowHandle"],
                    capture_output=True, text=True, timeout=5
                )
                handles = [h.strip() for h in result.stdout.strip().split('\n') if h.strip() and h.strip() != '0']
                
                if handles:
                    # Use PowerShell to bring window to front
                    subprocess.run(
                        ["powershell", "-Command",
                         f"""
                         Add-Type @'
                         using System;
                         using System.Runtime.InteropServices;
                         public class WinAPI {{
                             [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
                             [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
                         }}
'@
                         $h = [IntPtr]{handles[0]}
                         [WinAPI]::ShowWindow($h, 9)
                         [WinAPI]::SetForegroundWindow($h)
                         """],
                        capture_output=True, timeout=5
                    )
                    return {"ok": True, "output": f"Activated window: {app_name}"}
                else:
                    return {"ok": False, "error": f"No window found matching '{app_name}'"}
            except Exception as inner:
                return {"ok": False, "error": f"Could not find {app_name}: {repr(inner)}"}
        except Exception as e:
            return {"ok": False, "error": repr(e)}
>>>>>>> 3e64d789316217ee128862f1ededbace704ec132
