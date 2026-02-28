<<<<<<< HEAD
#!/usr/bin/env python3
"""
AUREON VISION SYSTEM
====================
Advanced computer vision for natural language GUI control.
Uses OCR to read screen text and find clickable elements by name.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

try:
    import pyautogui
    from PIL import Image
except ImportError:
    pyautogui = None
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class AureonVision:
    """
    Advanced vision system for AUREON.
    Can read screen text, find elements by name, and enable natural language GUI control.
    """
    
    def __init__(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui not installed: pip install pyautogui pillow")
        
        self.screen_size = pyautogui.size()
        self._last_screenshot = None
        self._last_ocr_data = None
        
        # Check for Tesseract OCR
        self.has_ocr = pytesseract is not None
        if not self.has_ocr:
            print("[WARN]?  pytesseract not installed - OCR unavailable (will use approximation)")
        else:
            print("[OK] OCR enabled - can find text on screen")
    
    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """Dispatch operation to appropriate handler"""
        handlers = {
            # Screen reading
            "screenshot": self.screenshot,
            "read_screen": self.read_screen,
            "describe_screen": self.describe_screen,
            
            # Text finding
            "find_text": self.find_text,
            "click_on_text": self.click_on_text,
            "find_and_click": self.find_and_click,
        }
        
        handler = handlers.get(op)
        if not handler:
            return {"ok": False, "error": f"unknown_operation: {op}"}
        
        try:
            return handler(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{op}_failed: {repr(e)}"}
    
    def screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Take screenshot"""
        if path is None:
            path = f"screenshot_{int(time.time())}.png"
        
        screenshot = pyautogui.screenshot()
        screenshot.save(path)
        self._last_screenshot = screenshot
        
        return {
            "ok": True,
            "path": str(Path(path).absolute()),
            "output": f"screenshot saved: {screenshot.width}x{screenshot.height}"
        }
    
    def read_screen(self) -> Dict[str, Any]:
        """Read all text from screen using OCR"""
        if not self.has_ocr:
            return {"ok": False, "error": "OCR not installed - pip install pytesseract"}
        
        screenshot = pyautogui.screenshot()
        text = pytesseract.image_to_string(screenshot)
        
        return {
            "ok": True,
            "text": text.strip(),
            "output": f"read {len(text)} characters"
        }
    
    def describe_screen(self) -> Dict[str, Any]:
        """Describe what's on screen"""
        result = self.read_screen()
        if not result.get("ok"):
            return {"ok": True, "output": "Cannot read screen - OCR not available"}
        
        text = result.get("text", "")
        lines = [l for l in text.split('\n') if l.strip()]
        
        return {
            "ok": True,
            "preview": text[:300],
            "lines": len(lines),
            "output": f"Screen has {len(lines)} lines of text"
        }
    
    def find_text(self, text: str) -> Dict[str, Any]:
        """Find text on screen and return position"""
        if not self.has_ocr:
            return {"ok": False, "error": "OCR not available"}
        
        screenshot = pyautogui.screenshot()
        ocr_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
        
        for i, word in enumerate(ocr_data['text']):
            if text.lower() in word.lower():
                x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                
                return {
                    "ok": True,
                    "found": True,
                    "position": {"x": x, "y": y},
                    "text": word,
                    "output": f"found '{word}' at ({x}, {y})"
                }
        
        return {"ok": False, "found": False, "output": f"'{text}' not found"}
    
    def click_on_text(self, text: str) -> Dict[str, Any]:
        """Find text and click on it"""
        result = self.find_text(text)
        if not result.get("found"):
            return result
        
        x = result["position"]["x"]
        y = result["position"]["y"]
        pyautogui.click(x, y)
        
        return {
            "ok": True,
            "text": text,
            "position": {"x": x, "y": y},
            "output": f"clicked '{text}' at ({x}, {y})"
        }
    
    def find_and_click(self, description: str) -> Dict[str, Any]:
        """Natural language click - extract key words and try to click"""
        words = [w for w in description.split() if len(w) > 2 and w.lower() not in ['the', 'a', 'an', 'button', 'tab']]
        
        for word in words:
            result = self.click_on_text(word)
            if result.get("ok"):
                return result
        
        return {"ok": False, "output": f"Could not find '{description}'"}

=======
#!/usr/bin/env python3
"""
AUREON VISION SYSTEM
====================
Advanced computer vision for natural language GUI control.
Uses OCR to read screen text and find clickable elements by name.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

try:
    import pyautogui
    from PIL import Image
except ImportError:
    pyautogui = None
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class AureonVision:
    """
    Advanced vision system for AUREON.
    Can read screen text, find elements by name, and enable natural language GUI control.
    """
    
    def __init__(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui not installed: pip install pyautogui pillow")
        
        self.screen_size = pyautogui.size()
        self._last_screenshot = None
        self._last_ocr_data = None
        
        # Check for Tesseract OCR
        self.has_ocr = pytesseract is not None
        if not self.has_ocr:
            print("[WARN]?  pytesseract not installed - OCR unavailable (will use approximation)")
        else:
            print("[OK] OCR enabled - can find text on screen")
    
    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """Dispatch operation to appropriate handler"""
        handlers = {
            # Screen reading
            "screenshot": self.screenshot,
            "read_screen": self.read_screen,
            "describe_screen": self.describe_screen,
            
            # Text finding
            "find_text": self.find_text,
            "click_on_text": self.click_on_text,
            "find_and_click": self.find_and_click,
        }
        
        handler = handlers.get(op)
        if not handler:
            return {"ok": False, "error": f"unknown_operation: {op}"}
        
        try:
            return handler(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{op}_failed: {repr(e)}"}
    
    def screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Take screenshot"""
        if path is None:
            path = f"screenshot_{int(time.time())}.png"
        
        screenshot = pyautogui.screenshot()
        screenshot.save(path)
        self._last_screenshot = screenshot
        
        return {
            "ok": True,
            "path": str(Path(path).absolute()),
            "output": f"screenshot saved: {screenshot.width}x{screenshot.height}"
        }
    
    def read_screen(self) -> Dict[str, Any]:
        """Read all text from screen using OCR"""
        if not self.has_ocr:
            return {"ok": False, "error": "OCR not installed - pip install pytesseract"}
        
        screenshot = pyautogui.screenshot()
        text = pytesseract.image_to_string(screenshot)
        
        return {
            "ok": True,
            "text": text.strip(),
            "output": f"read {len(text)} characters"
        }
    
    def describe_screen(self) -> Dict[str, Any]:
        """Describe what's on screen"""
        result = self.read_screen()
        if not result.get("ok"):
            return {"ok": True, "output": "Cannot read screen - OCR not available"}
        
        text = result.get("text", "")
        lines = [l for l in text.split('\n') if l.strip()]
        
        return {
            "ok": True,
            "preview": text[:300],
            "lines": len(lines),
            "output": f"Screen has {len(lines)} lines of text"
        }
    
    def find_text(self, text: str) -> Dict[str, Any]:
        """Find text on screen and return position"""
        if not self.has_ocr:
            return {"ok": False, "error": "OCR not available"}
        
        screenshot = pyautogui.screenshot()
        ocr_data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
        
        for i, word in enumerate(ocr_data['text']):
            if text.lower() in word.lower():
                x = ocr_data['left'][i] + ocr_data['width'][i] // 2
                y = ocr_data['top'][i] + ocr_data['height'][i] // 2
                
                return {
                    "ok": True,
                    "found": True,
                    "position": {"x": x, "y": y},
                    "text": word,
                    "output": f"found '{word}' at ({x}, {y})"
                }
        
        return {"ok": False, "found": False, "output": f"'{text}' not found"}
    
    def click_on_text(self, text: str) -> Dict[str, Any]:
        """Find text and click on it"""
        result = self.find_text(text)
        if not result.get("found"):
            return result
        
        x = result["position"]["x"]
        y = result["position"]["y"]
        pyautogui.click(x, y)
        
        return {
            "ok": True,
            "text": text,
            "position": {"x": x, "y": y},
            "output": f"clicked '{text}' at ({x}, {y})"
        }
    
    def find_and_click(self, description: str) -> Dict[str, Any]:
        """Natural language click - extract key words and try to click"""
        words = [w for w in description.split() if len(w) > 2 and w.lower() not in ['the', 'a', 'an', 'button', 'tab']]
        
        for word in words:
            result = self.click_on_text(word)
            if result.get("ok"):
                return result
        
        return {"ok": False, "output": f"Could not find '{description}'"}

>>>>>>> 3e64d789316217ee128862f1ededbace704ec132
