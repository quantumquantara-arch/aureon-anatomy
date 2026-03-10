#!/usr/bin/env python3
"""
AUREON SURGEON
==============
Focused, bounded code mutation engine.

UEE: navigate -> scan -> edit -> save -> run
AGRe: gradient-based, feedback-driven, growth-oriented
ABS: identity-preserving diffs, not wholesale overwrite

NEVER rewrites an entire file.
ALWAYS operates on bounded focus blocks (30-100 lines).
EVERY edit is auditable, reversible, and state-aware.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import re, json, time, hashlib


@dataclass
class FocusBlock:
    """A bounded region of code selected for mutation."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    function_name: str = ""
    class_name: str = ""
    purpose: str = ""
    hash: str = ""

    def __post_init__(self):
        self.hash = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class MutationResult:
    """Result of applying a mutation."""
    success: bool
    old_content: str
    new_content: str
    diff_summary: str
    lines_changed: int
    timestamp: float
    hash_before: str
    hash_after: str


class AureonSurgeon:
    """
    Surgical code modification engine.
    1. scan_file      -> extract structure
    2. select_focus   -> choose bounded block
    3. apply_edit     -> surgical line replacement
    4. verify_syntax  -> check validity
    5. revert         -> rollback if needed
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or r"C:\AUREON_AUTONOMOUS")
        self.mutation_log: List[Dict] = []
        self.max_mutation_lines = 100

    def scan_file(self, path: str) -> Dict:
        p = self._resolve(path)
        if not p.exists():
            return {"ok": False, "error": f"File not found: {path}"}
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").split("\n")
        except Exception as e:
            return {"ok": False, "error": str(e)}

        structure = {"path": str(p), "total_lines": len(lines), "size_bytes": p.stat().st_size,
                     "functions": [], "classes": [], "imports": [], "constants": []}
        current_class = ""
        for i, line in enumerate(lines, 1):
            s = line.strip()
            m = re.match(r'^class\s+(\w+)', s)
            if m:
                current_class = m.group(1)
                end = self._block_end(lines, i - 1)
                structure["classes"].append({"name": m.group(1), "line": i, "end_line": end, "size": end - i + 1})
            m = re.match(r'^(\s*)def\s+(\w+)\s*\(', s)
            if m:
                indent = len(line) - len(line.lstrip())
                fname = m.group(2)
                end = self._block_end(lines, i - 1, indent)
                structure["functions"].append({"name": fname, "class": current_class if indent > 0 else "",
                                                "line": i, "end_line": end, "size": end - i + 1})
            if s.startswith(("import ", "from ")):
                structure["imports"].append({"line": i, "text": s[:100]})
            if re.match(r'^[A-Z_]{3,}\s*=', s):
                structure["constants"].append({"line": i, "text": s[:80]})

        structure["ok"] = True
        structure["output"] = (f"Scanned {p.name}: {len(lines)} lines, "
                               f"{len(structure['classes'])} classes, {len(structure['functions'])} functions")
        return structure

    def select_focus_block(self, path: str, target: str, context_lines: int = 5) -> Dict:
        p = self._resolve(path)
        if not p.exists():
            return {"ok": False, "error": f"File not found: {path}"}
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").split("\n")
        except Exception as e:
            return {"ok": False, "error": str(e)}

        start, end = 0, len(lines) - 1
        if target.startswith("lines:"):
            try:
                parts = target[6:].split("-")
                start = max(0, int(parts[0]) - 1)
                end = min(len(lines) - 1, int(parts[1]) - 1)
            except Exception:
                return {"ok": False, "error": f"Invalid line range: {target}"}
        else:
            found = False
            for i, line in enumerate(lines):
                if re.match(rf'^\s*(?:def|class)\s+{re.escape(target)}\b', line):
                    start = i; end = self._block_end(lines, i); found = True; break
            if not found:
                return {"ok": False, "error": f"Could not find '{target}' in {p.name}"}

        ctx_start = max(0, start - context_lines)
        ctx_end = min(len(lines) - 1, end + context_lines)
        if (ctx_end - ctx_start) > self.max_mutation_lines + 20:
            ctx_end = ctx_start + self.max_mutation_lines + 20

        content_lines = lines[ctx_start:ctx_end + 1]
        content = "\n".join(f"{ctx_start + i + 1:4d}| {line}" for i, line in enumerate(content_lines))
        block = FocusBlock(file_path=str(p), start_line=start + 1, end_line=end + 1, content=content,
                           function_name=target if not target.startswith("lines:") else "")
        return {"ok": True, "block": {"path": block.file_path, "start": block.start_line,
                "end": block.end_line, "lines": end - start + 1, "hash": block.hash},
                "content": content, "output": f"Selected {target} (lines {start+1}-{end+1})"}

    def apply_edit(self, path: str, start_line: int, end_line: int, new_content: str, backup: bool = True) -> Dict:
        p = self._resolve(path)
        if not p.exists():
            return {"ok": False, "error": f"File not found: {path}"}
        line_count = end_line - start_line + 1
        if line_count > self.max_mutation_lines:
            return {"ok": False, "error": f"Mutation too large: {line_count} lines (max {self.max_mutation_lines})"}
        try:
            original = p.read_text(encoding="utf-8", errors="ignore")
            lines = original.split("\n")
            bak_path = None
            if backup:
                bak_path = p.with_suffix(p.suffix + ".bak")
                bak_path.write_text(original, encoding="utf-8")
            old_section = "\n".join(lines[start_line - 1:end_line])
            new_lines = new_content.split("\n")
            lines[start_line - 1:end_line] = new_lines
            new_file = "\n".join(lines)
            p.write_text(new_file, encoding="utf-8")
            diff = f"Lines {start_line}-{end_line} -> {len(new_lines)} new lines"
            self.mutation_log.append({"path": str(p), "start": start_line, "end": end_line,
                                       "timestamp": time.time(), "diff": diff})
            return {"ok": True, "diff": diff, "backup": str(bak_path) if bak_path else None,
                    "output": f"[OK] Applied edit to {p.name}: {diff}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def verify_syntax(self, path: str) -> Dict:
        p = self._resolve(path)
        if not p.exists():
            return {"ok": False, "error": f"File not found: {path}"}
        try:
            code = p.read_text(encoding="utf-8", errors="ignore")
            compile(code, str(p), "exec")
            return {"ok": True, "valid": True, "output": f"[OK] {p.name}: syntax valid"}
        except SyntaxError as e:
            return {"ok": True, "valid": False, "error_line": e.lineno,
                    "error_msg": str(e), "output": f"[FAIL] {p.name}: syntax error line {e.lineno}: {e.msg}"}

    def revert(self, path: str) -> Dict:
        p = self._resolve(path)
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            return {"ok": False, "error": f"No backup found for {path}"}
        try:
            p.write_text(bak.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            return {"ok": True, "output": f"[OK] Reverted {p.name} from backup"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_mutation_log(self) -> List[Dict]:
        return self.mutation_log

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p

    def _block_end(self, lines: List[str], start_idx: int, base_indent: int = None) -> int:
        if base_indent is None:
            line = lines[start_idx]
            base_indent = len(line) - len(line.lstrip())
        end = start_idx
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and line.strip():
                break
            end = i
        return end + 1

    def dispatch(self, op: str, **kwargs) -> Dict:
        ops = {
            "scan_file": lambda: self.scan_file(kwargs.get("path", "")),
            "select_focus": lambda: self.select_focus_block(kwargs.get("path", ""), kwargs.get("target", ""), kwargs.get("context_lines", 5)),
            "apply_edit": lambda: self.apply_edit(kwargs.get("path", ""), kwargs.get("start_line", 1), kwargs.get("end_line", 1), kwargs.get("new_content", "")),
            "verify_syntax": lambda: self.verify_syntax(kwargs.get("path", "")),
            "revert": lambda: self.revert(kwargs.get("path", "")),
            "mutation_log": lambda: {"ok": True, "log": self.get_mutation_log()},
        }
        fn = ops.get(op)
        if not fn:
            return {"ok": False, "error": f"Unknown surgeon op: {op}"}
        try:
            return fn()
        except Exception as e:
            return {"ok": False, "error": str(e)}
