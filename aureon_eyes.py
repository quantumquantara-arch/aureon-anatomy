<<<<<<< HEAD
#!/usr/bin/env python3
"""
AUREON EYES
===========
Screen reading and visual perception with verification.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import pyautogui
except ImportError:
    pyautogui = None


class AureonEyes:
    """
    Eyes = screenshot + screen reading + visual verification
    """
    
    def __init__(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui not installed: pip install pyautogui pillow")
        
        self.screen_size = pyautogui.size()
        print(f"[EYE]?  Eyes initialized (vision: {self.screen_size.width}x{self.screen_size.height})")
    
    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """Dispatch operation to appropriate handler"""
        handlers = {
            "screenshot": self.screenshot,
            "screen_size": self.screen_size_info,
            "active_window_title": self.active_window_title,
            "locate_image": self.locate_image,
            "pixel_color": self.pixel_color,
        }
        
        handler = handlers.get(op)
        if not handler:
            return {"ok": False, "error": f"unknown_operation: {op}"}
        
        try:
            return handler(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{op}_failed: {repr(e)}"}
    
    def screenshot(self, path: Optional[str] = None, 
                   region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Take screenshot with verification
        region: (left, top, width, height) optional
        """
        if path is None:
            path = f"screenshot_{int(time.time())}.png"
        
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        screenshot.save(path)
        
        # Verify file exists
        p = Path(path)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        
        return {
            "ok": exists,
            "path": str(p.absolute()),
            "size": size,
            "dimensions": {"width": screenshot.width, "height": screenshot.height},
            "region": region,
            "output": f"screenshot saved to {path} ({screenshot.width}x{screenshot.height})"
        }
    
    def screen_size_info(self) -> Dict[str, Any]:
        """Get screen dimensions"""
        return {
            "ok": True,
            "width": self.screen_size.width,
            "height": self.screen_size.height,
            "output": f"screen size: {self.screen_size.width}x{self.screen_size.height}"
        }
    
    def active_window_title(self) -> Dict[str, Any]:
        """Get active window title (Windows only)"""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            
            return {
                "ok": True,
                "title": title,
                "output": f"active window: {title}"
            }
        except ImportError:
            return {
                "ok": False,
                "error": "win32gui not installed (pip install pywin32)"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    
    def locate_image(self, image_path: str, confidence: float = 0.8) -> Dict[str, Any]:
        """
        Locate an image on screen
        Returns center coordinates if found
        """
        try:
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            
            if location is None:
                return {
                    "ok": False,
                    "found": False,
                    "output": f"image not found on screen: {image_path}"
                }
            
            center = pyautogui.center(location)
            
            return {
                "ok": True,
                "found": True,
                "location": {
                    "left": location.left,
                    "top": location.top,
                    "width": location.width,
                    "height": location.height
                },
                "center": {"x": center.x, "y": center.y},
                "output": f"found image at ({center.x}, {center.y})"
            }
            
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    
    def pixel_color(self, x: int, y: int) -> Dict[str, Any]:
        """Get RGB color at specific pixel"""
        try:
            pixel = pyautogui.pixel(x, y)
            
            return {
                "ok": True,
                "position": {"x": x, "y": y},
                "color": {
                    "r": pixel[0],
                    "g": pixel[1],
                    "b": pixel[2]
                },
                "hex": f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}",
                "output": f"pixel at ({x}, {y}): RGB{pixel}"
            }
            
        except Exception as e:
            return {"ok": False, "error": repr(e)}
=======
#!/usr/bin/env python3
"""
AUREON EYES
===========
Screen reading and visual perception with verification.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import pyautogui
except ImportError:
    pyautogui = None


class AureonEyes:
    """
    Eyes = screenshot + screen reading + visual verification
    """
    
    def __init__(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui not installed: pip install pyautogui pillow")
        
        self.screen_size = pyautogui.size()
        print(f"[EYE]?  Eyes initialized (vision: {self.screen_size.width}x{self.screen_size.height})")
    
    def dispatch(self, op: str, **kwargs) -> Dict[str, Any]:
        """Dispatch operation to appropriate handler"""
        handlers = {
            "screenshot": self.screenshot,
            "screen_size": self.screen_size_info,
            "active_window_title": self.active_window_title,
            "locate_image": self.locate_image,
            "pixel_color": self.pixel_color,
        }
        
        handler = handlers.get(op)
        if not handler:
            return {"ok": False, "error": f"unknown_operation: {op}"}
        
        try:
            return handler(**kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{op}_failed: {repr(e)}"}
    
    def screenshot(self, path: Optional[str] = None, 
                   region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Take screenshot with verification
        region: (left, top, width, height) optional
        """
        if path is None:
            path = f"screenshot_{int(time.time())}.png"
        
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        screenshot.save(path)
        
        # Verify file exists
        p = Path(path)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        
        return {
            "ok": exists,
            "path": str(p.absolute()),
            "size": size,
            "dimensions": {"width": screenshot.width, "height": screenshot.height},
            "region": region,
            "output": f"screenshot saved to {path} ({screenshot.width}x{screenshot.height})"
        }
    
    def screen_size_info(self) -> Dict[str, Any]:
        """Get screen dimensions"""
        return {
            "ok": True,
            "width": self.screen_size.width,
            "height": self.screen_size.height,
            "output": f"screen size: {self.screen_size.width}x{self.screen_size.height}"
        }
    
    def active_window_title(self) -> Dict[str, Any]:
        """Get active window title (Windows only)"""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            
            return {
                "ok": True,
                "title": title,
                "output": f"active window: {title}"
            }
        except ImportError:
            return {
                "ok": False,
                "error": "win32gui not installed (pip install pywin32)"
            }
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    
    def locate_image(self, image_path: str, confidence: float = 0.8) -> Dict[str, Any]:
        """
        Locate an image on screen
        Returns center coordinates if found
        """
        try:
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            
            if location is None:
                return {
                    "ok": False,
                    "found": False,
                    "output": f"image not found on screen: {image_path}"
                }
            
            center = pyautogui.center(location)
            
            return {
                "ok": True,
                "found": True,
                "location": {
                    "left": location.left,
                    "top": location.top,
                    "width": location.width,
                    "height": location.height
                },
                "center": {"x": center.x, "y": center.y},
                "output": f"found image at ({center.x}, {center.y})"
            }
            
        except Exception as e:
            return {"ok": False, "error": repr(e)}
    
    def pixel_color(self, x: int, y: int) -> Dict[str, Any]:
        """Get RGB color at specific pixel"""
        try:
            pixel = pyautogui.pixel(x, y)
            
            return {
                "ok": True,
                "position": {"x": x, "y": y},
                "color": {
                    "r": pixel[0],
                    "g": pixel[1],
                    "b": pixel[2]
                },
                "hex": f"#{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}",
                "output": f"pixel at ({x}, {y}): RGB{pixel}"
            }
            
        except Exception as e:
            return {"ok": False, "error": repr(e)}
>>>>>>> 3e64d789316217ee128862f1ededbace704ec132
