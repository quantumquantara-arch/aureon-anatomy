import os
import subprocess
import time
from typing import Optional, List
import psutil
import pyttsx3
import speech_recognition as sr
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeDriverManager

"""
AUREON HANDS - COMPLETE
=======================
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
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeDriverManager

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
            print("⚠  Selenium not installed: pip install selenium")
            print("   Browser control disabled. File operations still work.")

    # ?????????????????????????????????????????????????????????????????????
    # BROWSER CONNECTION (auto-retry)
    # ?????????????????????????????????????????????????????????????????????

    def _try_connect_browser(self) -> bool:
        """Try to connect to existing browser. Auto-launches Chrome if needed."""
        now = time.time()
        # Don't spam retries – wait at least 10 seconds between attempts
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
                        for attempt in range(15): # Up to 15 seconds
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
            options = ChromeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
            self.browser_connected = True
            print("✅ Connected to Chrome (port 9222)")
            return True
        except Exception as chrome_err:
            short_err = str(chrome_err).split('\n')[0][:120]
            print(f"   Chrome selenium failed: {short_err}")

        # Try Edge
        try:
            options = EdgeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Edge(service=EdgeService(EdgeDriverManager().install()), options=options)
            self.browser_connected = True
            print("✅ Connected to Edge (port 9222)")
            return True
        except Exception as edge_err:
            short_err = str(edge_err).split('\n')[0][:120]
            print(f"   Edge selenium failed: {short_err}")

        if not self.browser_connected:
            print("⚠  Browser launched but Selenium could not connect")
            print("   This usually means chromedriver version doesn't match Chrome version")
            print("   Update: pip install --upgrade selenium")
        return False

    def _ensure_browser(self) -> bool:
        """Ensure browser is connected, retry if necessary."""
        if self.browser_connected:
            try:
                _ = self.driver.title # check if driver is still alive
                return True
            except Exception:
                self.browser_connected = False
                self.driver = None
        
        if SELENIUM_AVAILABLE:
            return self._try_connect_browser()
        return False

    # ?????????????????????????????????????????????????????????????????????
    # FILE OPERATIONS
    # ?????????????????????????????????????????????????????????????????????

    def run_command(self, command: str, shell: str = "powershell") -> Dict[str, Any]:
        """
        Executes a shell command using PowerShell or cmd.
        Args:
            command (str): The command string to execute.
            shell (str): The shell to use ('powershell' or 'cmd'). Defaults to 'powershell'.
        Returns:
            Dict[str, Any]: A dictionary containing stdout, stderr, and returncode.
        """
        if shell.lower() == "powershell":
            cmd_prefix = ["powershell.exe", "-Command"]
        elif shell.lower() == "cmd":
            cmd_prefix = ["cmd.exe", "/C"]
        else:
            return {"ok": False, "error": f"Unsupported shell: {shell}. Use 'powershell' or 'cmd'."}

        try:
            # Use 'text=True' for universal newline support and string output
            process = subprocess.run(cmd_prefix + [command], capture_output=True, text=True, check=False)
            output = {
                "ok": True,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                "returncode": process.returncode,
                "output": process.stdout.strip() if process.stdout else process.stderr.strip()
            }
            if process.returncode != 0:
                output["ok"] = False
                output["error"] = f"Command failed with exit code {process.returncode}: {process.stderr.strip()}"
            return output
        except FileNotFoundError:
            return {"ok": False, "error": f"Shell executable not found: {shell}. Make sure it's in your PATH."}
        except Exception as e:
            return {"ok": False, "error": str(e)}
            
    def search_files(self, filename: str, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Searches for a file by name within the base directory or a specified path.
        Args:
            filename (str): The name of the file to search for.
            path (Optional[str]): The directory to start the search from. Defaults to self.base_dir.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status, and a list of 'found_files' or an 'error'.
        """
        search_path = Path(path) if path else self.base_dir
        found_files = []
        try:
            for root, _, files in os.walk(search_path):
                for file in files:
                    if filename == file:
                        found_files.append(str(Path(root) / file))
            return {"ok": True, "found_files": found_files, "output": f"Found {len(found_files)} files matching '{filename}' in '{search_path}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error searching for file: {e}"}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Reads the content of a specified file.
        Args:
            file_path (str): The full path to the file.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status, 'content' of the file, or an 'error'.
        """
        try:
            p = Path(file_path)
            if not p.is_file():
                return {"ok": False, "error": f"File not found: {file_path}", "output": f"Error: File not found at {file_path}"}
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return {"ok": True, "content": content, "output": f"Successfully read file: {file_path} (length: {len(content)})."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error reading file {file_path}: {e}"}

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Writes content to a specified file, creating it if it doesn't exist.
        Args:
            file_path (str): The full path to the file.
            content (str): The content to write to the file.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status or an 'error'.
        """
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
            return {"ok": True, "output": f"Successfully wrote to file: {file_path} (length: {len(content)})."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error writing to file {file_path}: {e}"}

    def list_files(self, path: Optional[str] = None, include_hidden: bool = False) -> Dict[str, Any]:
        """
        Lists files and directories in a given path.
        Args:
            path (Optional[str]): The directory to list. Defaults to self.base_dir.
            include_hidden (bool): Whether to include hidden files/directories.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status, lists of 'files' and 'directories', or an 'error'.
        """
        target_path = Path(path) if path else self.base_dir
        if not target_path.is_dir():
            return {"ok": False, "error": f"Path is not a directory or does not exist: {target_path}", "output": f"Error: Path is not a directory or does not exist: {target_path}"}

        files = []
        directories = []
        try:
            for item in target_path.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    directories.append(item.name)
            return {"ok": True, "files": files, "directories": directories, "output": f"Listing contents of {target_path}: {len(files)} files, {len(directories)} directories."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error listing files in {target_path}: {e}"}

    # ?????????????????????????????????????????????????????????????????????
    # BROWSER NAVIGATION & INTERACTION
    # ?????????????????????????????????????????????????????????????????????

    def go_to_url(self, url: str) -> Dict[str, Any]:
        """
        Navigates the browser to the specified URL.
        Args:
            url (str): The URL to navigate to.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            self.driver.get(url)
            return {"ok": True, "output": f"Navigated to {url}. Current URL: {self.driver.current_url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error navigating to {url}: {e}"}

    def click_on_text(self, text: str, exact_match: bool = True, timeout: int = 10) -> Dict[str, Any]:
        """
        Clicks on an element containing the specified text.
        Args:
            text (str): The text to search for on the page.
            exact_match (bool): If True, looks for exact text match. If False, looks for partial text.
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            if exact_match:
                # Try finding by exact link text, then by xpath for other elements
                try:
                    element = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, text)))
                except:
                    element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//*[normalize-space(text())='{text}']")))
            else:
                # Try finding by partial link text, then by xpath for other elements (contains)
                try:
                    element = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, text)))
                except:
                    element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//*[contains(normalize-space(text()),'{text}')]")))
            
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            time.sleep(0.5) # Give scroll time
            element.click()
            return {"ok": True, "output": f"Clicked on element with text: '{text}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error clicking on text '{text}': {e}"}

    def type_text(self, text: str, selector: str, by: str = "css", append: bool = False, timeout: int = 10) -> Dict[str, Any]:
        """
        Types text into a specified input field.
        Args:
            text (str): The text to type.
            selector (str): The CSS selector or XPath of the input field.
            by (str): The method to locate the element ('css' or 'xpath').
            append (bool): If True, append text; otherwise, clear and type.
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            by_method = By.CSS_SELECTOR if by.lower() == "css" else By.XPATH
            element = wait.until(EC.element_to_be_clickable((by_method, selector)))
            
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            time.sleep(0.5) # Give scroll time

            if not append:
                element.clear()
            element.send_keys(text)
            return {"ok": True, "output": f"Typed text into element '{selector}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error typing into element '{selector}': {e}"}

    def press(self, key: str) -> Dict[str, Any]:
        """
        Simulates pressing a special key (e.g., 'ENTER', 'ESCAPE').
        Args:
            key (str): The key to press (e.g., 'ENTER', 'ESCAPE', 'TAB'). Case-insensitive.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            # Map common key names to Selenium's Keys enum
            key_map = {
                'ENTER': Keys.ENTER,
                'RETURN': Keys.RETURN,
                'ESCAPE': Keys.ESCAPE,
                'ESC': Keys.ESCAPE,
                'TAB': Keys.TAB,
                'SPACE': Keys.SPACE,
                'BACKSPACE': Keys.BACKSPACE,
                'DELETE': Keys.DELETE,
                'ARROW_UP': Keys.ARROW_UP,
                'UP': Keys.ARROW_UP,
                'ARROW_DOWN': Keys.ARROW_DOWN,
                'DOWN': Keys.ARROW_DOWN,
                'ARROW_LEFT': Keys.ARROW_LEFT,
                'LEFT': Keys.ARROW_LEFT,
                'ARROW_RIGHT': Keys.ARROW_RIGHT,
                'RIGHT': Keys.ARROW_RIGHT,
            }
            selenium_key = key_map.get(key.upper())
            if not selenium_key:
                return {"ok": False, "error": f"Unsupported key: {key}. Supported: {list(key_map.keys())}"}
            
            ActionChains(self.driver).send_keys(selenium_key).perform()
            return {"ok": True, "output": f"Pressed key: '{key}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error pressing key '{key}': {e}"}

    def get_page_source(self) -> Dict[str, Any]:
        """
        Returns the full HTML source of the current page.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'html' content.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            return {"ok": True, "html": self.driver.page_source, "output": "Retrieved page source."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting page source: {e}"}

    def get_current_url(self) -> Dict[str, Any]:
        """
        Returns the current URL of the browser.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'url'.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            return {"ok": True, "url": self.driver.current_url, "output": f"Current URL: {self.driver.current_url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting current URL: {e}"}

    def get_title(self) -> Dict[str, Any]:
        """
        Returns the title of the current page.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'title'.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            return {"ok": True, "title": self.driver.title, "output": f"Page title: {self.driver.title}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting page title: {e}"}

    def get_text_from_element(self, selector: str, by: str = "css", timeout: int = 10) -> Dict[str, Any]:
        """
        Extracts visible text from an element specified by a selector.
        Args:
            selector (str): The CSS selector or XPath of the element.
            by (str): The method to locate the element ('css' or 'xpath').
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'text' content.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            by_method = By.CSS_SELECTOR if by.lower() == "css" else By.XPATH
            element = wait.until(EC.presence_of_element_located((by_method, selector)))
            text_content = element.text
            return {"ok": True, "text": text_content, "output": f"Extracted text from element '{selector}': {text_content[:100]}..."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error extracting text from element '{selector}': {e}"}

    def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Opens a new browser tab. Optionally navigates to a URL.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            if url:
                self.driver.get(url)
            return {"ok": True, "output": f"Opened new tab. Current URL: {self.driver.current_url if url else 'about:blank'}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error opening new tab: {e}"}

    def switch_tab(self, tab_index: int) -> Dict[str, Any]:
        """
        Switches to a browser tab by its index (0 for the first tab).
        Args:
            tab_index (int): The zero-based index of the tab to switch to.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            window_handles = self.driver.window_handles
            if tab_index < 0 or tab_index >= len(window_handles):
                return {"ok": False, "error": f"Tab index {tab_index} out of range. Current tabs: {len(window_handles)}"}
            self.driver.switch_to.window(window_handles[tab_index])
            return {"ok": True, "output": f"Switched to tab index {tab_index}. Current URL: {self.driver.current_url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error switching to tab {tab_index}: {e}"}

    def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """
        Scrolls the current page up or down.
        Args:
            direction (str): 'up' or 'down'. Defaults to 'down'.
            amount (int): Number of pixels to scroll. Defaults to 500.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            if direction.lower() == "down":
                self.driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction.lower() == "up":
                self.driver.execute_script(f"window.scrollBy(0, -{amount});")
            else:
                return {"ok": False, "error": "Invalid scroll direction. Use 'up' or 'down'."}
            return {"ok": True, "output": f"Scrolled {direction} by {amount} pixels."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error scrolling {direction}: {e}"}

    def get_element_screenshot(self, selector: str, by: str = "css", path: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
        """
        Takes a screenshot of a specific element.
        Args:
            selector (str): The CSS selector or XPath of the element.
            by (str): The method to locate the element ('css' or 'xpath').
            path (Optional[str]): File path to save the screenshot. Defaults to a unique filename in current dir.
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: Path to the screenshot or error.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            by_method = By.CSS_SELECTOR if by.lower() == "css" else By.XPATH
            element = wait.until(EC.presence_of_element_located((by_method, selector)))

            if path is None:
                path = f"element_screenshot_{int(time.time())}.png"
            
            element.screenshot(path)
            return {"ok": True, "path": str(Path(path).absolute()), "output": f"Screenshot of element '{selector}' saved to {path}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error taking element screenshot for '{selector}': {e}"}
    
    def get_all_links(self) -> Dict[str, Any]:
        """
        Retrieves all visible links (href and text) from the current page.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and a list of 'links'.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            # Execute JavaScript to get all link hrefs and visible text
            links = self.driver.execute_script(
                """
                var links = [];
                var elements = document.querySelectorAll('a[href]');
                for (var i = 0; i < elements.length; i++) {
                    var link = elements[i];
                    var text = link.innerText.trim();
                    if (text) { // Only include links with visible text
                        links.push({
                            href: link.href,
                            text: text
                        });
                    }
                }
                return links;
                """
            )
            return {"ok": True, "links": links, "output": f"Found {len(links)} visible links."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting all links: {e}"}

    def close_browser(self) -> Dict[str, Any]:
        """
        Closes the browser.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if self.driver:
            try:
                self.driver.quit()
                self.browser_connected = False
                self.driver = None
                return {"ok": True, "output": "Browser closed."}
            except Exception as e:
                return {"ok": False, "error": str(e), "output": f"Error closing browser: {e}"}
        return {"ok": True, "output": "No browser was active to close."}

    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """
        Dispatches operation to appropriate handler.
        """
        handlers = {
            # File operations
            "run_command": self.run_command,
            "search_files": self.search_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,

            # Browser operations
            "go_to_url": self.go_to_url,
            "click_on_text": self.click_on_text,
            "type_text": self.type_text,
            "press": self.press,
            "get_page_source": self.get_page_source,
            "get_current_url": self.get_current_url,
            "get_title": self.get_title,
            "get_text_from_element": self.get_text_from_element,
            "new_tab": self.new_tab,
            "switch_tab": self.switch_tab,
            "scroll": self.scroll,
            "get_element_screenshot": self.get_element_screenshot,
            "get_all_links": self.get_all_links,
            "close_browser": self.close_browser,
        }

        handler = handlers.get(op)
        if not handler:
            return {"ok": False, "error": f"Unknown operation: {op}"}

        try:
            return handler(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{op}_failed: {repr(e)}"}

# Example Usage (for testing purposes, if run directly)
if __name__ == "__main__":
    hands = AureonHands()
    
    print("\n--- File Operations Test ---")
    # Test run_command
    result = hands.run_command("dir", shell="cmd")
    print(f"run_command (cmd): {result['output'][:200]}...")

    result = hands.run_command("Get-ChildItem -Path C:\", shell="powershell")
    print(f"run_command (powershell): {result['output'][:200]}...")

    # Test write_file and read_file
    test_content = "Hello, Aureon! This is a test file."
    write_result = hands.write_file("test_file.txt", test_content)
    print(f"write_file: {write_result['output']}")
    read_result = hands.read_file("test_file.txt")
    print(f"read_file: {read_result['output']}")

    # Test list_files
    list_result = hands.list_files()
    print(f"list_files: {list_result['output']}")

    if hands.browser_connected:
        print("\n--- Browser Operations Test ---")
        # Test go_to_url
        go_result = hands.go_to_url("https://www.google.com")
        print(f"go_to_url: {go_result['output']}")
        time.sleep(2)

        # Test type_text and press (search on Google)
        type_result = hands.type_text("Aureon AI", "textarea[name='q']")
        print(f"type_text: {type_result['output']}")
        press_result = hands.press("ENTER")
        print(f"press: {press_result['output']}")
        time.sleep(3)

        # Test get_title
        title_result = hands.get_title()
        print(f"get_title: {title_result['output']}")

        # Test scroll
        scroll_result = hands.scroll("down", 1000)
        print(f"scroll: {scroll_result['output']}")
        time.sleep(1)

        # Test get_all_links
        links_result = hands.get_all_links()
        print(f"get_all_links: Found {len(links_result.get('links', []))} links.")
        if links_result['links']:
            print(f"First link: {links_result['links'][0]}")

        # Test new_tab and switch_tab
        new_tab_result = hands.new_tab("https://www.wikipedia.org")
        print(f"new_tab: {new_tab_result['output']}")
        time.sleep(2)
        switch_tab_result = hands.switch_tab(0) # Go back to Google
        print(f"switch_tab: {switch_tab_result['output']}")
        time.sleep(2)
        switch_tab_result = hands.switch_tab(1) # Go back to Wikipedia
        print(f"switch_tab: {switch_tab_result['output']}")
        time.sleep(2)

        # Test get_element_screenshot
        screenshot_result = hands.get_element_screenshot("img[alt='Wikipedia']")
        print(f"get_element_screenshot: {screenshot_result['output']}")

        # Test close_browser
        close_result = hands.close_browser()
        print(f"close_browser: {close_result['output']}")
    else:
        print("\n--- Skipping Browser Operations (not connected) ---")

    print("\n--- All tests completed ---")
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
from webdriver_manager.microsoft import EdgeDriverManager # Added for Edge

"""
AUREON HANDS - COMPLETE
=======================
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
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.edge.service import Service as EdgeService # Added EdgeService
    from selenium.webdriver.edge.options import Options as EdgeOptions # Added EdgeOptions
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.microsoft import EdgeDriverManager # Added EdgeDriverManager

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

    # ?????????????????????????????????????????????????????????????????????
    # BROWSER CONNECTION (auto-retry)
    # ?????????????????????????????????????????????????????????????????????

    def _try_connect_browser(self) -> bool:
        """Try to connect to existing browser. Auto-launches Chrome if needed."""
        now = time.time()
        # Don't spam retries – wait at least 10 seconds between attempts
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
            edge_paths = [ # Added Edge paths
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
                        for attempt in range(15): # Up to 15 seconds
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
            options = ChromeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
            self.browser_connected = True
            print("\u2705 Connected to Chrome (port 9222)")
            return True
        except Exception as chrome_err:
            short_err = str(chrome_err).split('\n')[0][:120]
            print(f"   Chrome selenium failed: {short_err}")

        # Try Edge # Added Edge connection logic
        try:
            options = EdgeOptions()
            options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
            self.driver = webdriver.Edge(service=EdgeService(EdgeDriverManager().install()), options=options)
            self.browser_connected = True
            print("\u2705 Connected to Edge (port 9222)")
            return True
        except Exception as edge_err:
            short_err = str(edge_err).split('\n')[0][:120]
            print(f"   Edge selenium failed: {short_err}")

        if not self.browser_connected:
            print("\u26A0  Browser launched but Selenium could not connect")
            print("   This usually means chromedriver version doesn't match Chrome version")
            print("   Update: pip install --upgrade selenium")
        return False

    def _ensure_browser(self) -> bool:
        """Ensure browser is connected, retry if necessary."""
        if self.browser_connected:
            try:
                _ = self.driver.title # check if driver is still alive
                return True
            except Exception:
                self.browser_connected = False
                self.driver = None
        
        if SELENIUM_AVAILABLE:
            return self._try_connect_browser()
        return False

    # ?????????????????????????????????????????????????????????????????????
    # FILE OPERATIONS
    # ?????????????????????????????????????????????????????????????????????

    def run_command(self, command: str, shell: str = "powershell") -> Dict[str, Any]:
        """
        Executes a shell command using PowerShell or cmd.
        Args:
            command (str): The command string to execute.
            shell (str): The shell to use ('powershell' or 'cmd'). Defaults to 'powershell'.
        Returns:
            Dict[str, Any]: A dictionary containing stdout, stderr, and returncode.
        """
        if shell.lower() == "powershell":
            cmd_prefix = ["powershell.exe", "-Command"]
        elif shell.lower() == "cmd":
            cmd_prefix = ["cmd.exe", "/C"]
        else:
            return {"ok": False, "error": f"Unsupported shell: {shell}. Use 'powershell' or 'cmd'."}

        try:
            # Use 'text=True' for universal newline support and string output
            process = subprocess.run(cmd_prefix + [command], capture_output=True, text=True, check=False)
            output = {
                "ok": True,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                "returncode": process.returncode,
                "output": process.stdout.strip() if process.stdout else process.stderr.strip()
            }
            if process.returncode != 0:
                output["ok"] = False
                output["error"] = f"Command failed with exit code {process.returncode}: {process.stderr.strip()}"
            return output
        except FileNotFoundError:
            return {"ok": False, "error": f"Shell executable not found: {shell}. Make sure it's in your PATH."}
        except Exception as e:
            return {"ok": False, "error": str(e)}
            
    def search_files(self, filename: str, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Searches for a file by name within the base directory or a specified path.
        Args:
            filename (str): The name of the file to search for.
            path (Optional[str]): The directory to start the search from. Defaults to self.base_dir.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status, and a list of 'found_files' or an 'error'.
        """
        search_path = Path(path) if path else self.base_dir
        found_files = []
        try:
            for root, _, files in os.walk(search_path):
                for file in files:
                    if filename == file:
                        found_files.append(str(Path(root) / file))
            return {"ok": True, "found_files": found_files, "output": f"Found {len(found_files)} files matching '{filename}' in '{search_path}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error searching for file: {e}"}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Reads the content of a specified file.
        Args:
            file_path (str): The full path to the file.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status, 'content' of the file, or an 'error'.
        """
        try:
            p = Path(file_path)
            if not p.is_file():
                return {"ok": False, "error": f"File not found: {file_path}", "output": f"Error: File not found at {file_path}"}
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return {"ok": True, "content": content, "output": f"Successfully read file: {file_path} (length: {len(content)})."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error reading file {file_path}: {e}"}

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Writes content to a specified file, creating it if it doesn't exist.
        Args:
            file_path (str): The full path to the file.
            content (str): The content to write to the file.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status or an 'error'.
        """
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
            return {"ok": True, "output": f"Successfully wrote to file: {file_path} (length: {len(content)})."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error writing to file {file_path}: {e}"}

    def list_files(self, path: Optional[str] = None, include_hidden: bool = False) -> Dict[str, Any]:
        """
        Lists files and directories in a given path.
        Args:
            path (Optional[str]): The directory to list. Defaults to self.base_dir.
            include_hidden (bool): Whether to include hidden files/directories.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status, lists of 'files' and 'directories', or an 'error'.
        """
        target_path = Path(path) if path else self.base_dir
        if not target_path.is_dir():
            return {"ok": False, "error": f"Path is not a directory or does not exist: {target_path}", "output": f"Error: Path is not a directory or does not exist: {target_path}"}

        files = []
        directories = []
        try:
            for item in target_path.iterdir():
                if not include_hidden and item.name.startswith('.'):
                    continue
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    directories.append(item.name)
            return {"ok": True, "files": files, "directories": directories, "output": f"Listing contents of {target_path}: {len(files)} files, {len(directories)} directories."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error listing files in {target_path}: {e}"}

    # ?????????????????????????????????????????????????????????????????????
    # BROWSER NAVIGATION & INTERACTION
    # ?????????????????????????????????????????????????????????????????????

    def go_to_url(self, url: str) -> Dict[str, Any]:
        """
        Navigates the browser to the specified URL.
        Args:
            url (str): The URL to navigate to.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            self.driver.get(url)
            return {"ok": True, "output": f"Navigated to {url}. Current URL: {self.driver.current_url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error navigating to {url}: {e}"}

    def click_on_text(self, text: str, exact_match: bool = True, timeout: int = 10) -> Dict[str, Any]:
        """
        Clicks on an element containing the specified text.
        Args:
            text (str): The text to search for on the page.
            exact_match (bool): If True, looks for exact text match. If False, looks for partial text.
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            if exact_match:
                # Try finding by exact link text, then by xpath for other elements
                try:
                    element = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, text)))
                except:
                    element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//*[normalize-space(text())='{text}']")))
            else:
                # Try finding by partial link text, then by xpath for other elements (contains)
                try:
                    element = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, text)))
                except:
                    element = wait.until(EC.element_to_be_clickable((By.XPATH, f"//*[contains(normalize-space(text()),'{text}')]")))
            
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            time.sleep(0.5) # Give scroll time
            element.click()
            return {"ok": True, "output": f"Clicked on element with text: '{text}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error clicking on text '{text}': {e}"}

    def type_text(self, text: str, selector: str, by: str = "css", append: bool = False, timeout: int = 10) -> Dict[str, Any]:
        """
        Types text into a specified input field.
        Args:
            text (str): The text to type.
            selector (str): The CSS selector or XPath of the input field.
            by (str): The method to locate the element ('css' or 'xpath').
            append (bool): If True, append text; otherwise, clear and type.
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            by_method = By.CSS_SELECTOR if by.lower() == "css" else By.XPATH
            element = wait.until(EC.element_to_be_clickable((by_method, selector)))
            
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            time.sleep(0.5) # Give scroll time

            if not append:
                element.clear()
            element.send_keys(text)
            return {"ok": True, "output": f"Typed text into element '{selector}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error typing into element '{selector}': {e}"}

    def press(self, key: str) -> Dict[str, Any]:
        """
        Simulates pressing a special key (e.g., 'ENTER', 'ESCAPE').
        Args:
            key (str): The key to press (e.g., 'ENTER', 'ESCAPE', 'TAB'). Case-insensitive.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            # Map common key names to Selenium's Keys enum
            key_map = {
                'ENTER': Keys.ENTER,
                'RETURN': Keys.RETURN,
                'ESCAPE': Keys.ESCAPE,
                'ESC': Keys.ESCAPE,
                'TAB': Keys.TAB,
                'SPACE': Keys.SPACE,
                'BACKSPACE': Keys.BACKSPACE,
                'DELETE': Keys.DELETE,
                'ARROW_UP': Keys.ARROW_UP,
                'UP': Keys.ARROW_UP,
                'ARROW_DOWN': Keys.ARROW_DOWN,
                'DOWN': Keys.ARROW_DOWN,
                'ARROW_LEFT': Keys.ARROW_LEFT,
                'LEFT': Keys.ARROW_LEFT,
                'ARROW_RIGHT': Keys.ARROW_RIGHT,
                'RIGHT': Keys.ARROW_RIGHT,
            }
            selenium_key = key_map.get(key.upper())
            if not selenium_key:
                return {"ok": False, "error": f"Unsupported key: {key}. Supported: {list(key_map.keys())}"}
            
            ActionChains(self.driver).send_keys(selenium_key).perform()
            return {"ok": True, "output": f"Pressed key: '{key}'."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error pressing key '{key}': {e}"}

    def get_page_source(self) -> Dict[str, Any]:
        """
        Returns the full HTML source of the current page.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'html' content.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            return {"ok": True, "html": self.driver.page_source, "output": "Retrieved page source."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting page source: {e}"}

    def get_current_url(self) -> Dict[str, Any]:
        """
        Returns the current URL of the browser.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'url'.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            return {"ok": True, "url": self.driver.current_url, "output": f"Current URL: {self.driver.current_url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting current URL: {e}"}

    def get_title(self) -> Dict[str, Any]:
        """
        Returns the title of the current page.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'title'.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            return {"ok": True, "title": self.driver.title, "output": f"Page title: {self.driver.title}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting page title: {e}"}

    def get_text_from_element(self, selector: str, by: str = "css", timeout: int = 10) -> Dict[str, Any]:
        """
        Extracts visible text from an element specified by a selector.
        Args:
            selector (str): The CSS selector or XPath of the element.
            by (str): The method to locate the element ('css' or 'xpath').
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and 'text' content.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            by_method = By.CSS_SELECTOR if by.lower() == "css" else By.XPATH
            element = wait.until(EC.presence_of_element_located((by_method, selector)))
            text_content = element.text
            return {"ok": True, "text": text_content, "output": f"Extracted text from element '{selector}': {text_content[:100]}..."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error extracting text from element '{selector}': {e}"}

    def new_tab(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Opens a new browser tab. Optionally navigates to a URL.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            self.driver.execute_script("window.open('');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            if url:
                self.driver.get(url)
            return {"ok": True, "output": f"Opened new tab. Current URL: {self.driver.current_url if url else 'about:blank'}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error opening new tab: {e}"}

    def switch_tab(self, tab_index: int) -> Dict[str, Any]:
        """
        Switches to a browser tab by its index (0 for the first tab).
        Args:
            tab_index (int): The zero-based index of the tab to switch to.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            window_handles = self.driver.window_handles
            if tab_index < 0 or tab_index >= len(window_handles):
                return {"ok": False, "error": f"Tab index {tab_index} out of range. Current tabs: {len(window_handles)}"}
            self.driver.switch_to.window(window_handles[tab_index])
            return {"ok": True, "output": f"Switched to tab index {tab_index}. Current URL: {self.driver.current_url}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error switching to tab {tab_index}: {e}"}

    def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """
        Scrolls the current page up or down.
        Args:
            direction (str): 'up' or 'down'. Defaults to 'down'.
            amount (int): Number of pixels to scroll. Defaults to 500.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            if direction.lower() == "down":
                self.driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction.lower() == "up":
                self.driver.execute_script(f"window.scrollBy(0, -{amount});")
            else:
                return {"ok": False, "error": "Invalid scroll direction. Use 'up' or 'down'."}
            return {"ok": True, "output": f"Scrolled {direction} by {amount} pixels."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error scrolling {direction}: {e}"}

    def get_element_screenshot(self, selector: str, by: str = "css", path: Optional[str] = None, timeout: int = 10) -> Dict[str, Any]:
        """
        Takes a screenshot of a specific element.
        Args:
            selector (str): The CSS selector or XPath of the element.
            by (str): The method to locate the element ('css' or 'xpath').
            path (Optional[str]): File path to save the screenshot. Defaults to a unique filename in current dir.
            timeout (int): Maximum time to wait for the element to be present.
        Returns:
            Dict[str, Any]: Path to the screenshot or error.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            wait = WebDriverWait(self.driver, timeout)
            by_method = By.CSS_SELECTOR if by.lower() == "css" else By.XPATH
            element = wait.until(EC.presence_of_element_located((by_method, selector)))

            if path is None:
                path = f"element_screenshot_{int(time.time())}.png"
            
            element.screenshot(path)
            return {"ok": True, "path": str(Path(path).absolute()), "output": f"Screenshot of element '{selector}' saved to {path}"}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error taking element screenshot for '{selector}': {e}"}
    
    def get_all_links(self) -> Dict[str, Any]:
        """
        Retrieves all visible links (href and text) from the current page.
        Returns:
            Dict[str, Any]: A dictionary with 'ok' status and a list of 'links'.
        """
        if not self._ensure_browser():
            return {"ok": False, "error": "Browser not connected."}
        try:
            # Execute JavaScript to get all link hrefs and visible text
            links = self.driver.execute_script(
                """
                var links = [];
                var elements = document.querySelectorAll('a[href]');
                for (var i = 0; i < elements.length; i++) {
                    var link = elements[i];
                    var text = link.innerText.trim();
                    if (text) { // Only include links with visible text
                        links.push({
                            href: link.href,
                            text: text
                        });
                    }
                }
                return links;
                """
            )
            return {"ok": True, "links": links, "output": f"Found {len(links)} visible links."}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error getting all links: {e}"}

    def close_browser(self) -> Dict[str, Any]:
        """
        Closes the browser.
        Returns:
            Dict[str, Any]: Status of the operation.
        """
        if self.driver:
            try:
                self.driver.quit()
                self.browser_connected = False
                self.driver = None
                return {"ok": True, "output": "Browser closed."}
            except Exception as e:
                return {"ok": False, "error": str(e), "output": f"Error closing browser: {e}"}
        return {"ok": True, "output": "No browser was active to close."}

    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """
        Dispatches operation to appropriate handler.
        """
        handlers = {
            # File operations
            "run_command": self.run_command,
            "search_files": self.search_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,

            # Browser operations
            "go_to_url": self.go_to_url,
            "click_on_text": self.click_on_text,
            "type_text": self.type_text,
            "press": self.press,
            "get_page_source": self.get_page_source,
            "get_current_url": self.get_current_url,
            "get_title": self.get_title,
            "get_text_from_element": self.get_text_from_element,
            "new_tab": self.new_tab,
            "switch_tab": self.switch_tab,
            "scroll": self.scroll,
            "get_element_screenshot": self.get_element_screenshot,
            "get_all_links": self.get_all_links,
            "close_browser": self.close_browser,
        }

        handler = handlers.get(op)
        if not handler:
            return {"ok": False, "error": f"Unknown operation: {op}"}

        try:
            return handler(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{op}_failed: {repr(e)}"}

# Example Usage (for testing purposes, if run directly)
if __name__ == "__main__":
    hands = AureonHands()
    
    print("\n--- File Operations Test ---")
    # Test run_command
    result = hands.run_command("dir", shell="cmd")
    print(f"run_command (cmd): {result['output'][:200]}...")

    result = hands.run_command("Get-ChildItem -Path C:\", shell="powershell")
    print(f"run_command (powershell): {result['output'][:200]}...")

    # Test write_file and read_file
    test_content = "Hello, Aureon! This is a test file."
    write_result = hands.write_file("test_file.txt", test_content)
    print(f"write_file: {write_result['output']}")
    read_result = hands.read_file("test_file.txt")
    print(f"read_file: {read_result['output']}")

    # Test list_files
    list_result = hands.list_files()
    print(f"list_files: {list_result['output']}")

    if hands.browser_connected:
        print("\n--- Browser Operations Test ---")
        # Test go_to_url
        go_result = hands.go_to_url("https://www.google.com")
        print(f"go_to_url: {go_result['output']}")
        time.sleep(2)

        # Test type_text and press (search on Google)
        type_result = hands.type_text("Aureon AI", "textarea[name='q']")
        print(f"type_text: {type_result['output']}")
        press_result = hands.press("ENTER")
        print(f"press: {press_result['output']}")
        time.sleep(3)

        # Test get_title
        title_result = hands.get_title()
        print(f"get_title: {title_result['output']}")

        # Test scroll
        scroll_result = hands.scroll("down", 1000)
        print(f"scroll: {scroll_result['output']}")
        time.sleep(1)

        # Test get_all_links
        links_result = hands.get_all_links()
        print(f"get_all_links: Found {len(links_result.get('links', []))} links.")
        if links_result['links']:
            print(f"First link: {links_result['links'][0]}")

        # Test new_tab and switch_tab
        new_tab_result = hands.new_tab("https://www.wikipedia.org")
        print(f"new_tab: {new_tab_result['output']}")
        time.sleep(2)
        switch_tab_result = hands.switch_tab(0) # Go back to Google
        print(f"switch_tab: {switch_tab_result['output']}")
        time.sleep(2)
        switch_tab_result = hands.switch_tab(1) # Go back to Wikipedia
        print(f"switch_tab: {switch_tab_result['output']}")
        time.sleep(2)

        # Test get_element_screenshot
        screenshot_result = hands.get_element_screenshot("img[alt='Wikipedia']")
        print(f"get_element_screenshot: {screenshot_result['output']}")

        # Test close_browser
        close_result = hands.close_browser()
        print(f"close_browser: {close_result['output']}")
    else:
        print("\n--- Skipping Browser Operations (not connected) ---")

    print("\n--- All tests completed ---")
