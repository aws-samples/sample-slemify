// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package report

// HTMLTemplate is the self-contained HTML report template.
// The report Job generates JSON data and injects it into this template.
const HTMLTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Slemify Report — {{PROJECT}}</title>
<style>
:root {
  --bg: #0f1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --yellow: #d29922; --red: #f85149;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
header { display: flex; align-items: center; gap: 16px; padding: 24px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
header h1 { font-size: 24px; font-weight: 600; }
header .badge { padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.badge-pass { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }
.badge-warn { background: rgba(210,153,34,0.15); color: var(--yellow); border: 1px solid rgba(210,153,34,0.3); }
.badge-fail { background: rgba(248,81,73,0.15); color: var(--red); border: 1px solid rgba(248,81,73,0.3); }
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; overflow-x: auto; }
.tab { padding: 10px 20px; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; font-size: 14px; white-space: nowrap; transition: all 0.2s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.panel { display: none; }
.panel.active { display: block; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.card h3 { font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.metric { display: inline-block; margin-right: 32px; margin-bottom: 8px; }
.metric .value { font-size: 28px; font-weight: 700; }
.metric .label { font-size: 12px; color: var(--muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; padding: 8px 12px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
tr:hover { background: rgba(88,166,255,0.04); }
.bar { height: 8px; border-radius: 4px; background: var(--border); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.summary { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-bottom: 24px; font-size: 15px; line-height: 1.8; }
.summary p { margin-bottom: 12px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }
.tag-slm { background: rgba(88,166,255,0.15); color: var(--accent); }
.tag-llm { background: rgba(210,153,34,0.15); color: var(--yellow); }
footer { text-align: center; padding: 24px 0; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); margin-top: 32px; }
</style>
</head>
<body>
<div class="container">
<header>
  <h1>Slemify Report</h1>
  <span class="badge {{VERDICT_CLASS}}">{{VERDICT}}</span>
  <span style="color:var(--muted);font-size:14px">{{PROJECT}} — {{TIMESTAMP}}</span>
</header>

<div class="tabs">
  <div class="tab active" onclick="showTab('overview')">Overview</div>
  <div class="tab" onclick="showTab('data')">Data</div>
  <div class="tab" onclick="showTab('training')">Training</div>
  <div class="tab" onclick="showTab('accuracy')">Accuracy</div>
  <div class="tab" onclick="showTab('comparison')">SLM vs LLM</div>
  <div class="tab" onclick="showTab('cost')">Cost & Performance</div>
</div>

<div id="overview" class="panel active">
  <div class="summary">{{SUMMARY}}</div>
  <div class="grid">
    <div class="card">
      <h3>Accuracy</h3>
      <div class="metric"><span class="value" style="color:{{ACC_COLOR}}">{{ACCURACY}}%</span><br><span class="label">overall accuracy</span></div>
    </div>
    <div class="card">
      <h3>Latency</h3>
      <div class="metric"><span class="value">{{SLM_P50}}ms</span><br><span class="label">SLM p50</span></div>
      <div class="metric"><span class="value" style="color:var(--muted)">{{LLM_P50}}ms</span><br><span class="label">LLM p50</span></div>
    </div>
    <div class="card">
      <h3>Monthly Cost (100K req/day)</h3>
      <div class="metric"><span class="value" style="color:var(--green)">${{SLM_MONTHLY}}</span><br><span class="label">SLM (fixed)</span></div>
      <div class="metric"><span class="value" style="color:var(--muted)">${{LLM_MONTHLY}}</span><br><span class="label">LLM API</span></div>
    </div>
  </div>
</div>

<div id="data" class="panel">{{DATA_CONTENT}}</div>
<div id="training" class="panel">{{TRAINING_CONTENT}}</div>
<div id="accuracy" class="panel">{{ACCURACY_CONTENT}}</div>
<div id="comparison" class="panel">{{COMPARISON_CONTENT}}</div>
<div id="cost" class="panel">{{COST_CONTENT}}</div>

<footer>Generated by Slemify — {{TIMESTAMP}}</footer>
</div>

<script>
function showTab(id) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}
</script>
</body>
</html>`
