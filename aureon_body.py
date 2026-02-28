"""
AUREON BODY - COMPLETE LAPTOP CONTROL
Mouse, keyboard, window management - NO SURVEILLANCE
"""

import pyautogui
try:
    import pygetwindow as gw
    PYGETWINDOW_OK = True
except:
    PYGETWINDOW_OK = False

import psutil
import subprocess

class AureonBody:
    """Pure laptop control for AUREON to use"""
    
    def __init__(self):
        pyautogui.FAILSAFE = False
    
    def click_at(self, x, y):
        """Click at screen coordinates"""
        pyautogui.click(x, y)
    
    def right_click_at(self, x, y):
        """Right click at coordinates"""
        pyautogui.rightClick(x, y)
    
    def type_text(self, text, interval=0.05):
        """Type text with keyboard"""
        pyautogui.write(text, interval=interval)
    
    def press_key(self, key):
        """Press a single key"""
        pyautogui.press(key)
    
    def hotkey(self, *keys):
        """Press hotkey combination"""
        pyautogui.hotkey(*keys)
    
    def open_application(self, app_path):
        """Launch application"""
        subprocess.Popen(app_path)
    
    def get_active_window(self):
        """Get currently active window"""
        if PYGETWINDOW_OK:
            try:
                return gw.getActiveWindow()
            except:
                return None
        return None
    
    def get_mouse_position(self):
        """Get current mouse coordinates"""
        return pyautogui.position()
    
    def move_mouse(self, x, y, duration=0.5):
        """Move mouse smoothly"""
        pyautogui.moveTo(x, y, duration=duration)
    
    def execute(self, action):
        """Parse and execute action command"""
        action_lower = action.lower()
        
        if 'click' in action_lower:
            try:
                parts = action_lower.split('at')[1].strip().split(',')
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                self.click_at(x, y)
                return f"Clicked at ({x}, {y})"
            except:
                return "Could not parse click coordinates"
        
        elif 'type' in action_lower:
            try:
                text = action.split('type', 1)[1].strip()
                self.type_text(text)
                return f"Typed: {text}"
            except:
                return "Could not parse text to type"
        
        elif 'press' in action_lower:
            try:
                key = action_lower.split('press')[1].strip()
                self.press_key(key)
                return f"Pressed: {key}"
            except:
                return "Could not parse key to press"
        
        return f"Unknown action: {action}"
