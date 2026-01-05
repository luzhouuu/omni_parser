#!/usr/bin/env python3
"""Wanfang Pipeline Web Server.

A unified web interface for the complete Wanfang medical literature pipeline:
Search → Download → Classify

Usage:
    python scripts/wanfang_server.py --port 8080

Then open: http://localhost:8080
"""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

load_dotenv()

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def is_relative_to(path: Path, base: Path) -> bool:
    """Check if path is relative to base (Python 3.8 compatible)."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def json_bytes(obj: Any) -> bytes:
    """Serialize object to JSON bytes."""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def which(cmd: str) -> str | None:
    """Find executable in PATH."""
    import shutil
    return shutil.which(cmd)


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PipelineTask:
    """Represents a running pipeline task."""
    task_id: str
    status: str  # pending, searching, downloading, classifying, completed, error
    created_at: str
    query: str
    start_year: str
    end_year: str
    max_articles: int
    resource_type: str  # chinese, foreign, all
    drug_keywords: list[str]

    # Progress tracking
    current_step: str = ""
    progress: float = 0.0
    log_messages: list[str] = field(default_factory=list)

    # Results
    search_count: int = 0  # New articles to download
    search_total: int = 0  # Total search results (including already downloaded)
    download_count: int = 0
    classify_count: int = 0
    error_message: str = ""

    # Process handle (not serialized)
    _process: subprocess.Popen | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "query": self.query,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "max_articles": self.max_articles,
            "resource_type": self.resource_type,
            "drug_keywords": self.drug_keywords,
            "current_step": self.current_step,
            "progress": self.progress,
            "log_messages": self.log_messages[-50:],  # Last 50 messages
            "search_count": self.search_count,
            "search_total": self.search_total,
            "download_count": self.download_count,
            "classify_count": self.classify_count,
            "error_message": self.error_message,
        }


class TaskManager:
    """Manages pipeline tasks."""

    def __init__(self):
        self.tasks: dict[str, PipelineTask] = {}
        self.current_task_id: str | None = None
        self._lock = threading.Lock()

    def create_task(
        self,
        query: str,
        start_year: str,
        end_year: str,
        max_articles: int,
        resource_type: str,
        drug_keywords: list[str],
    ) -> PipelineTask:
        """Create a new pipeline task."""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = PipelineTask(
            task_id=task_id,
            status="pending",
            created_at=utc_now_iso(),
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_articles=max_articles,
            resource_type=resource_type,
            drug_keywords=drug_keywords,
        )
        with self._lock:
            self.tasks[task_id] = task
            self.current_task_id = task_id
        return task

    def get_current_task(self) -> PipelineTask | None:
        """Get the current running task."""
        with self._lock:
            if self.current_task_id:
                return self.tasks.get(self.current_task_id)
        return None

    def get_task(self, task_id: str) -> PipelineTask | None:
        """Get task by ID."""
        with self._lock:
            return self.tasks.get(task_id)

    def log(self, task_id: str, message: str):
        """Add log message to task."""
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                timestamp = datetime.now().strftime("%H:%M:%S")
                task.log_messages.append(f"[{timestamp}] {message}")


# Global task manager
task_manager = TaskManager()


def run_pipeline_async(task: PipelineTask):
    """Run the pipeline in a background thread."""
    try:
        # Step 1: Search
        task.status = "searching"
        task.current_step = "搜索文献..."
        task.progress = 0.1
        task_manager.log(task.task_id, f"开始搜索: {task.query}")

        search_cmd = [
            sys.executable, str(SCRIPTS_DIR / "wanfang_search.py"),
            "--query", task.query,
            "--start-year", task.start_year,
            "--end-year", task.end_year,
            "--resource-type", task.resource_type,
            "--export",
            "--no-stay",
        ]

        task_manager.log(task.task_id, f"执行命令: {' '.join(search_cmd[:4])}...")

        proc = subprocess.Popen(
            search_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        task._process = proc

        # Read output line by line
        total_found = 0
        new_articles = 0
        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line:
                task_manager.log(task.task_id, line)
                # Parse search output
                import re
                # "Total results: 185"
                if "Total results:" in line:
                    match = re.search(r'Total results:\s*(\d+)', line)
                    if match:
                        total_found = int(match.group(1))
                # "Found 10 new articles (not in history)"
                elif "new articles" in line.lower():
                    match = re.search(r'(\d+)\s*new articles', line, re.IGNORECASE)
                    if match:
                        new_articles = int(match.group(1))
                # "New articles to download: 10"
                elif "New articles to download:" in line:
                    match = re.search(r'New articles to download:\s*(\d+)', line)
                    if match:
                        new_articles = int(match.group(1))

        proc.wait()
        if proc.returncode != 0:
            task.status = "error"
            task.error_message = f"Search failed with code {proc.returncode}"
            task_manager.log(task.task_id, f"搜索失败: {task.error_message}")
            return

        # Set search counts
        task.search_total = total_found
        task.search_count = new_articles if new_articles > 0 else total_found

        # Double check with pending_download.csv
        pending_csv = DATA_DIR / "pending_download.csv"
        if pending_csv.exists():
            with open(pending_csv, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) > 0:
                    task.search_count = len(rows)

        if total_found > 0 and new_articles == 0:
            task_manager.log(task.task_id, f"搜索完成，共 {total_found} 篇（全部已下载过）")
        elif total_found > new_articles > 0:
            task_manager.log(task.task_id, f"搜索完成，共 {total_found} 篇，{new_articles} 篇新文献")
        else:
            task_manager.log(task.task_id, f"搜索完成，找到 {task.search_count} 篇文献")
        task.progress = 0.3

        # Step 2: Download
        task.status = "downloading"
        task.current_step = "下载 PDF..."
        task_manager.log(task.task_id, "开始下载文献...")

        max_download = min(task.max_articles, task.search_count) if task.max_articles > 0 else task.search_count

        # Get Wanfang credentials from env
        wanfang_user = os.getenv("WANFANG_USERNAME", "")
        wanfang_pass = os.getenv("WANFANG_PASSWORD", "")

        download_cmd = [
            sys.executable, str(SCRIPTS_DIR / "wanfang_download.py"),
            "--max-papers", str(max_download),
        ]
        if wanfang_user and wanfang_pass:
            download_cmd.extend(["--username", wanfang_user, "--password", wanfang_pass])

        proc = subprocess.Popen(
            download_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        task._process = proc

        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line:
                task_manager.log(task.task_id, line)
                # Try to extract progress
                if "✅" in line or "downloaded" in line.lower():
                    task.download_count += 1
                    task.progress = 0.3 + 0.4 * (task.download_count / max(max_download, 1))

        proc.wait()

        # Count actual downloads
        if PAPERS_DIR.exists():
            task.download_count = len(list(PAPERS_DIR.glob("*.pdf")))

        task_manager.log(task.task_id, f"下载完成，共 {task.download_count} 篇")
        task.progress = 0.7

        # Step 3: Classify
        task.status = "classifying"
        task.current_step = "分类文献..."
        task_manager.log(task.task_id, "开始分类文献...")

        classify_cmd = [
            sys.executable, str(SCRIPTS_DIR / "wanfang_classify.py"),
        ]

        if task.drug_keywords:
            classify_cmd.extend(["--drugs", ",".join(task.drug_keywords)])

        proc = subprocess.Popen(
            classify_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        task._process = proc

        for line in iter(proc.stdout.readline, ''):
            line = line.strip()
            if line:
                task_manager.log(task.task_id, line)
                if "✅" in line or "ICSR" in line or "Rejection" in line:
                    task.classify_count += 1
                    task.progress = 0.7 + 0.3 * (task.classify_count / max(task.download_count, 1))

        proc.wait()

        # Done
        task.status = "completed"
        task.current_step = "完成"
        task.progress = 1.0
        task_manager.log(task.task_id, f"Pipeline 完成！搜索:{task.search_count} 下载:{task.download_count} 分类:{task.classify_count}")

    except Exception as e:
        task.status = "error"
        task.error_message = str(e)
        task_manager.log(task.task_id, f"Pipeline 错误: {e}")


class AppHandler(BaseHTTPRequestHandler):
    server_version = "WanfangPipeline/1.0"

    def log_message(self, format: str, *args) -> None:
        """Override to suppress default logging."""
        pass

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path in {"/", "/index.html"}:
                return self._serve_static_file(STATIC_DIR / "index.html")

            if path in {"/classify", "/classify.html"}:
                return self._serve_static_file(STATIC_DIR / "classify.html")

            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                file_path = (STATIC_DIR / rel).resolve()
                if not is_relative_to(file_path, STATIC_DIR):
                    raise ApiError(HTTPStatus.NOT_FOUND, "not_found")
                return self._serve_static_file(file_path)

            if path == "/api/health":
                return self._handle_health()

            if path == "/api/status":
                return self._handle_status()

            if path == "/api/results":
                return self._handle_results()

            raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

        except ApiError as e:
            self._send_json(e.status, {"error": e.message})

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)

            if parsed.path == "/api/pipeline":
                return self._handle_pipeline()

            if parsed.path == "/api/stop":
                return self._handle_stop()

            if parsed.path == "/api/classify":
                return self._handle_classify()

            raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

        except ApiError as e:
            self._send_json(e.status, {"error": e.message})

    def _handle_health(self) -> None:
        """Return system health status."""
        payload = {
            "status": "ok",
            "utc_time": utc_now_iso(),
            "env": {
                "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
                "wanfang_username": bool(os.getenv("WANFANG_USERNAME")),
                "llm_model": os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
                "classify_model": os.getenv("CLASSIFY_MODEL_NAME", "gpt-4o"),
            },
            "tools": {
                "pdftotext": bool(which("pdftotext")),
                "playwright": True,
            },
            "paths": {
                "data_dir": str(DATA_DIR),
                "papers_dir": str(PAPERS_DIR),
                "papers_count": len(list(PAPERS_DIR.glob("*.pdf"))) if PAPERS_DIR.exists() else 0,
            },
        }
        self._send_json(HTTPStatus.OK, payload)

    def _handle_status(self) -> None:
        """Return current task status."""
        task = task_manager.get_current_task()
        if task:
            payload = task.to_dict()
        else:
            payload = {
                "status": "idle",
                "message": "No active task",
            }
        self._send_json(HTTPStatus.OK, payload)

    def _handle_results(self) -> None:
        """Return classification results."""
        results_path = DATA_DIR / "classification_results.csv"

        if not results_path.exists():
            return self._send_json(HTTPStatus.OK, {"results": [], "count": 0})

        results = []
        with open(results_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

        self._send_json(HTTPStatus.OK, {
            "results": results,
            "count": len(results),
            "file": str(results_path),
        })

    def _handle_pipeline(self) -> None:
        """Start a new pipeline task."""
        # Check if task is already running
        current = task_manager.get_current_task()
        if current and current.status not in {"completed", "error"}:
            raise ApiError(HTTPStatus.CONFLICT, "task_already_running")

        # Parse request body
        body = self._read_body(max_bytes=10 * 1024)
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_json")

        # Extract parameters
        query = (data.get("query") or "").strip()
        if not query:
            raise ApiError(HTTPStatus.BAD_REQUEST, "missing_query")

        start_year = str(data.get("start_year", "2020"))
        end_year = str(data.get("end_year", datetime.now().year))
        max_articles = int(data.get("max_articles", 0))
        resource_type = data.get("resource_type", "chinese")
        if resource_type not in ("chinese", "foreign", "all"):
            resource_type = "chinese"

        drug_keywords = data.get("drug_keywords", [])
        if isinstance(drug_keywords, str):
            drug_keywords = [k.strip() for k in drug_keywords.split(",") if k.strip()]

        # Create and start task
        task = task_manager.create_task(
            query=query,
            start_year=start_year,
            end_year=end_year,
            max_articles=max_articles,
            resource_type=resource_type,
            drug_keywords=drug_keywords,
        )

        # Run in background thread
        thread = threading.Thread(target=run_pipeline_async, args=(task,), daemon=True)
        thread.start()

        self._send_json(HTTPStatus.OK, {
            "task_id": task.task_id,
            "status": "started",
            "message": "Pipeline started",
        })

    def _handle_stop(self) -> None:
        """Stop the current task."""
        task = task_manager.get_current_task()
        if not task:
            raise ApiError(HTTPStatus.NOT_FOUND, "no_active_task")

        if task._process:
            try:
                task._process.terminate()
                task._process.wait(timeout=5)
            except:
                task._process.kill()

        task.status = "error"
        task.error_message = "Task stopped by user"
        task_manager.log(task.task_id, "Task stopped by user")

        self._send_json(HTTPStatus.OK, {"status": "stopped"})

    def _handle_classify(self) -> None:
        """Handle file upload and classification."""
        import tempfile
        import re

        # Parse query parameters
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        max_pages = int(params.get("max_pages", ["50"])[0])
        drug_keywords_str = params.get("drug_keywords", [""])[0]
        drug_keywords = [k.strip() for k in drug_keywords_str.split(",") if k.strip()]

        # Parse multipart form data
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ApiError(HTTPStatus.BAD_REQUEST, "expected_multipart_form")

        # Extract boundary
        boundary_match = re.search(r'boundary=(.+?)(?:;|$)', content_type)
        if not boundary_match:
            raise ApiError(HTTPStatus.BAD_REQUEST, "no_boundary")
        boundary = boundary_match.group(1).strip('"')

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 100 * 1024 * 1024:  # 100MB limit
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "file_too_large")

        body = self.rfile.read(content_length)

        # Parse multipart
        files = []
        parts = body.split(f"--{boundary}".encode())

        for part in parts:
            if b"Content-Disposition" not in part:
                continue

            # Extract headers and content
            if b"\r\n\r\n" in part:
                header_section, content = part.split(b"\r\n\r\n", 1)
            else:
                continue

            # Extract filename
            header_str = header_section.decode("utf-8", errors="ignore")
            filename_match = re.search(r'filename="([^"]+)"', header_str)
            if not filename_match:
                continue

            filename = filename_match.group(1)

            # Remove trailing boundary markers
            if content.endswith(b"\r\n"):
                content = content[:-2]
            if content.endswith(b"--"):
                content = content[:-2]
            if content.endswith(b"\r\n"):
                content = content[:-2]

            files.append((filename, content))

        if not files:
            raise ApiError(HTTPStatus.BAD_REQUEST, "no_files_uploaded")

        # Import classification module
        from wanfang_classify import extract_pdf_text, classify_with_openai

        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for filename, content in files:
                # Save file temporarily
                file_path = tmpdir_path / filename
                file_path.write_bytes(content)

                # Process based on file type
                ext = file_path.suffix.lower()

                if ext == ".pdf":
                    text, method = extract_pdf_text(file_path, max_pages)
                elif ext == ".txt":
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    method = "txt"
                else:
                    text = ""
                    method = "unsupported"

                if not text.strip():
                    results.append({
                        "filename": filename,
                        "label": "Error",
                        "confidence": 0,
                        "has_drug": False,
                        "has_ae": False,
                        "has_causality": False,
                        "has_special_situation": False,
                        "patient_mode": "unknown",
                        "patient_max_n": None,
                        "extract_method": method,
                        "needs_review": True,
                        "error": "Could not extract text",
                    })
                    continue

                # Classify
                result = classify_with_openai(text, filename, drug_keywords)
                result.extract_method = method

                # Convert to dict
                from dataclasses import asdict
                result_dict = asdict(result)

                # Convert evidence lists to strings
                for key in ["drug_evidence", "ae_evidence", "causality_evidence", "special_evidence", "patient_evidence"]:
                    if result_dict.get(key):
                        result_dict[key] = "; ".join(result_dict[key])

                results.append(result_dict)

        self._send_json(HTTPStatus.OK, {"results": results, "count": len(results)})

    def _serve_static_file(self, path: Path) -> None:
        """Serve a static file."""
        if not path.exists() or not path.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

        data = path.read_bytes()
        ctype, _ = mimetypes.guess_type(str(path))
        if not ctype:
            ctype = "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: object) -> None:
        """Send JSON response."""
        data = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self, *, max_bytes: int) -> bytes:
        """Read request body."""
        raw_len = self.headers.get("Content-Length")
        if not raw_len:
            return b""
        try:
            length = int(raw_len)
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid_content_length")
        if length <= 0:
            return b""
        if length > max_bytes:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body_too_large")
        return self.rfile.read(length)


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the HTTP server."""
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Check for static files
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        print(f"Warning: {index_path} not found. Frontend won't work.")

    httpd = ThreadingHTTPServer((host, port), AppHandler)

    print("=" * 60)
    print("Wanfang Pipeline Server")
    print("=" * 60)
    print(f"Server: http://{host}:{port}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Papers dir: {PAPERS_DIR}")
    print(f"OpenAI API: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
    print(f"Wanfang login: {'✓' if os.getenv('WANFANG_USERNAME') else '✗'}")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wanfang Pipeline Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
