"""ui.html: the flow map draws only the steps and models the active
configuration runs, for workshop rows 0 to 4 and the decision-model gate.

Serves ui.html from a local stub server (127.0.0.1, free port) whose /config
and /query return what server.py sends for one configuration, then drives it
with headless Chromium. Skipped when Playwright is not installed.
Set UI_MAP_SCREENSHOT_DIR to save a screenshot of each map.
"""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests import _path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

with open(os.path.join(_path.DEMO_DIR, "ui.html"), encoding="utf-8") as _f:
    UI_HTML = _f.read().encode()

QUESTION = "What does the minValues field do in a Karpenter v1 NodePool requirement?"

# Step names as agent/graph.py emits them for the minValues question.
TRIAGE_CPU = ["Triage classifier \u00b7 ONNX Runtime (CPU)", "Intent router (CPU)"]
RETRIEVE_HOSTED = ["Embedder (hosted, Bedrock)", "OpenSearch (vector DB)", "Reranker (off)"]
RETRIEVE_TUNED = ["Embedder (tuned encoder, CPU)", "OpenSearch (vector DB)", "Reranker (cross-encoder, CPU)"]

# name -> (steps, step names emitted, {visible node id: classes}, label overrides)
CONFIGS = {
    "row0": ({"triage": "off", "embed": "bedrock", "rerank": "off", "analyst": "llm", "gate": "off"},
             RETRIEVE_HOSTED + ["Analyst \u00b7 LLM (Bedrock)"],
             {"retrieve": {"llm"}, "generate": {"llm"}},
             {"generate": "Analyst LLM"}),
    "row1": ({"triage": "classifier", "embed": "bedrock", "rerank": "off", "analyst": "llm", "gate": "off"},
             TRIAGE_CPU + RETRIEVE_HOSTED + ["Analyst \u00b7 LLM (Bedrock)"],
             {"triage": {"cpu", "tuned"}, "plan": {"cpu"}, "lint": {"cpu"}, "tools": {"cpu"},
              "retrieve": {"llm"}, "generate": {"llm"}},
             {"generate": "Analyst LLM"}),
    "row2": ({"triage": "classifier", "embed": "slemify", "rerank": "on", "analyst": "llm", "gate": "off"},
             TRIAGE_CPU + RETRIEVE_TUNED + ["Analyst \u00b7 LLM (Bedrock)"],
             {"triage": {"cpu", "tuned"}, "plan": {"cpu"}, "lint": {"cpu"}, "tools": {"cpu"},
              "retrieve": {"cpu", "tuned"}, "rerank": {"cpu"}, "generate": {"llm"}},
             {"generate": "Analyst LLM"}),
    "row3": ({"triage": "classifier", "embed": "slemify", "rerank": "on", "analyst": "slm", "gate": "off"},
             TRIAGE_CPU + RETRIEVE_TUNED + ["Analyst SLM (CPU)"],
             {"triage": {"cpu", "tuned"}, "plan": {"cpu"}, "lint": {"cpu"}, "tools": {"cpu"},
              "retrieve": {"cpu", "tuned"}, "rerank": {"cpu"}, "generate": {"cpu"}},
             {"generate": "Analyst SLM"}),
    "row4": ({"triage": "classifier", "embed": "slemify", "rerank": "on", "analyst": "slm", "gate": "llm"},
             TRIAGE_CPU + RETRIEVE_TUNED + ["Analyst SLM (CPU)", "Faithfulness gate (LLM)"],
             {"triage": {"cpu", "tuned"}, "plan": {"cpu"}, "lint": {"cpu"}, "tools": {"cpu"},
              "retrieve": {"cpu", "tuned"}, "rerank": {"cpu"}, "generate": {"cpu"},
              "llm": {"llm"}, "critic": {"llm"}, "remediate": {"cpu"}},
             {"generate": "Analyst SLM", "critic": "Gate"}),
    "decision": ({"triage": "classifier", "embed": "slemify", "rerank": "on", "analyst": "slm", "gate": "decision"},
                 TRIAGE_CPU + RETRIEVE_TUNED + ["Analyst SLM (CPU)", "Faithfulness gate \u00b7 decision model (CPU)"],
                 {"triage": {"cpu", "tuned"}, "plan": {"cpu"}, "lint": {"cpu"}, "tools": {"cpu"},
                  "retrieve": {"cpu", "tuned"}, "rerank": {"cpu"}, "generate": {"cpu"},
                  "llm": {"llm"}, "critic": {"cpu"}, "remediate": {"cpu"}},
                 {"generate": "Analyst SLM", "critic": "Gate (decision model)"}),
}
# Nodes a step lights for the minValues question (no manifest, no tools,
# gate accepts, nothing to remediate).
LIT = {"triage", "plan", "retrieve", "rerank", "generate", "critic"}
ALL_NODES = ["triage", "plan", "lint", "tools", "retrieve", "rerank", "generate", "llm", "critic", "remediate"]


def sse_body(steps, names):
    ev = [{"type": "steps", "steps": steps}]
    for n in names:
        ev.append({"type": "step_start", "name": n, "note": ""})
        ev.append({"type": "step_done", "name": n, "ms": 10, "detail": ""})
    ev.append({"type": "token", "text": "answer"})
    ev.append({"type": "cost", "usd": 0.01, "bedrock_calls": 1, "steps": steps, "total_ms": 100})
    ev.append({"type": "total", "ms": 100})
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in ev) + "data: [DONE]\n\n").encode()


def error_body(steps):
    """What server.py sends when the agent graph raises after one step."""
    ev = [{"type": "steps", "steps": steps},
          {"type": "step_start", "name": "Analyst SLM (CPU)", "note": ""},
          {"type": "error", "message": "RuntimeError: boom"},
          {"type": "cost", "usd": 0.0, "bedrock_calls": 0, "steps": steps, "total_ms": 5},
          {"type": "total", "ms": 5}]
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in ev) + "data: [DONE]\n\n").encode()
class Stub:
    steps = None
    names = None
    body = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/config":
            self.send(json.dumps({"apply_enabled": False, "steps": Stub.steps}).encode(), "application/json")
        else:
            self.send(UI_HTML, "text/html")

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send(Stub.body or sse_body(Stub.steps, Stub.names), "text/event-stream")


@unittest.skipIf(sync_playwright is None, "playwright is not installed")
class UiMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}/"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def map_state(self, page):
        return page.evaluate("""ids => ids.map(id => {
            const el = document.getElementById('g-' + id);
            return {id, visible: el.offsetParent !== null, cls: [...el.classList],
                    label: el.querySelector('.glbl').textContent};
        })""", ALL_NODES)

    def check(self, name):
        steps, names, expected, labels = CONFIGS[name]
        Stub.steps, Stub.names = steps, names
        page = self.browser.new_page()
        try:
            # The markdown renderer comes from a CDN; the test does not need it.
            page.route("https://cdn.jsdelivr.net/**",
                       lambda r: r.fulfill(body="", content_type="application/javascript"))
            page.goto(self.url)
            # At page load the map follows /config: wait until it shows exactly
            # the expected nodes (times out, and fails, if it never does).
            page.wait_for_function(
                """([ids, want]) => ids.filter(id => document.getElementById('g-' + id).offsetParent !== null)
                       .sort().join() === want""",
                arg=[ALL_NODES, ",".join(sorted(expected))], timeout=5000)
            page.fill("#input", QUESTION)
            page.click("#send")
            page.wait_for_function("!document.getElementById('send').disabled")
            state = self.map_state(page)
            if os.environ.get("UI_MAP_SCREENSHOT_DIR"):
                page.locator(".trace-pane").screenshot(
                    path=os.path.join(os.environ["UI_MAP_SCREENSHOT_DIR"], f"ui-map-stub-{name}.png"))
        finally:
            page.close()
        visible = {s["id"]: s for s in state if s["visible"]}
        self.assertEqual(set(visible), set(expected), name)
        for node, cls in expected.items():
            got = set(visible[node]["cls"]) & {"cpu", "llm", "tuned"}
            self.assertEqual(got, cls, f"{name}: {node}")
        for node, label in labels.items():
            self.assertEqual(visible[node]["label"], label, f"{name}: {node} label")
        done = {s["id"] for s in state if "done" in s["cls"]}
        self.assertEqual(done, LIT & set(expected), f"{name}: lit nodes")
        self.assertTrue(done <= set(visible), f"{name}: a hidden node is lit")

    def test_row0_monolith(self):
        self.check("row0")

    def test_row1_triage(self):
        self.check("row1")

    def test_row2_retrieval(self):
        self.check("row2")

    def test_row3_analyst(self):
        self.check("row3")

    def test_row4_gate(self):
        self.check("row4")

    def test_gate_decision(self):
        self.check("decision")
    def test_error_event_shows_message(self):
        Stub.steps, Stub.names = CONFIGS["row4"][0], []
        Stub.body = error_body(Stub.steps)
        page = self.browser.new_page()
        try:
            page.route("https://cdn.jsdelivr.net/**",
                       lambda r: r.fulfill(body="", content_type="application/javascript"))
            page.goto(self.url)
            page.fill("#input", QUESTION)
            page.click("#send")
            page.wait_for_function("!document.getElementById('send').disabled")
            alert = page.locator(".msg.error[role=alert]")
            self.assertEqual(alert.count(), 1)
            text = alert.inner_text()
            tally = page.locator("#tallyText").inner_text()
        finally:
            page.close()
            Stub.body = None
        self.assertIn("The query failed.", text)
        self.assertIn("RuntimeError: boom", text)
        # A failed query does not count as a run handled on CPU.
        self.assertNotIn("/ 1", tally)


if __name__ == "__main__":
    unittest.main()
