import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from workbench import app


def test_existing_web_assets_environment_and_job_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RUNS_DIR", str(tmp_path / "not-created"))
    monkeypatch.setattr(app, "JOBS", {})
    import queue
    monkeypatch.setattr(app, "JOB_QUEUE", queue.Queue())
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = "http://127.0.0.1:%d" % server.server_address[1]
    try:
        with urlopen(root, timeout=10) as response:
            assert b"plotly.min.js" in response.read()
        with urlopen(root + "/vendor/plotly.min.js", timeout=10) as response:
            assert b"plotly.js v2.35.0" in response.read(200)
        with urlopen(root + "/api/env", timeout=10) as response:
            assert json.load(response)["torch"]
        with urlopen(root + "/api/runs", timeout=10) as response:
            assert json.load(response) == {"runs": []}
        request = Request(root + "/api/jobs", data=json.dumps({"type": "solve", "params": {}}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            job_id = json.load(response)["job_id"]
        assert app.JOB_QUEUE.get_nowait() == job_id
        with urlopen(root + "/api/jobs/" + job_id, timeout=10) as response:
            assert json.load(response)["status"] == "queued"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=10)
