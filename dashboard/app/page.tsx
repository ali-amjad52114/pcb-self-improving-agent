"use client";

import { useMemo, useState } from "react";

type Run = {
  id: string;
  mode: "MEMORY" | "COLD";
  time: string;
  status: "complete" | "running" | "failed";
  f1: string;
  action: string;
  delta: string;
};

const initialRuns: Run[] = [
  { id: "Run #12", mode: "MEMORY", time: "16:17:12", status: "complete", f1: ".710", action: "Class weights", delta: "−.004" },
  { id: "Run #11", mode: "COLD", time: "16:08:44", status: "complete", f1: ".749", action: "Increase epochs", delta: "+.039" },
  { id: "Run #10", mode: "MEMORY", time: "15:56:31", status: "failed", f1: ".665", action: "Augmentation", delta: "−.045" },
  { id: "Run #09", mode: "MEMORY", time: "15:42:05", status: "complete", f1: ".706", action: "Class weights", delta: "−.004" },
];

const stages = [
  { key: "data", eyebrow: "REAL DATA", title: "PCB dataset", note: "2,953 defect crops", tone: "cyan" },
  { key: "train", eyebrow: "BASELINE", title: "Train classifier", note: "NumPy image model", tone: "blue" },
  { key: "eval", eyebrow: "MEASURE", title: "Evaluate", note: "Macro F1 · .710", tone: "purple" },
  { key: "memory", eyebrow: "ATLAS", title: "Retrieve memory", note: "5 relevant lessons", tone: "green" },
  { key: "science", eyebrow: "FIREWORKS", title: "Scientist", note: "Change class weights", tone: "orange" },
  { key: "experiment", eyebrow: "EXPERIMENT", title: "Retrain + compare", note: "Delta · −.004", tone: "yellow" },
  { key: "critic", eyebrow: "CRITIC", title: "Extract lesson", note: "Failed result stored", tone: "red" },
];

const classMetrics = [
  ["Missing hole", 0.79],
  ["Mouse bite", 0.43],
  ["Open circuit", 0.69],
  ["Short", 0.82],
  ["Spur", 0.76],
  ["Spurious copper", 0.77],
] as const;

export default function Home() {
  const [runs, setRuns] = useState(initialRuns);
  const [selected, setSelected] = useState("Run #12");
  const [memoryMode, setMemoryMode] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [activeStage, setActiveStage] = useState("memory");
  const [zoom, setZoom] = useState(1);

  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selected) ?? runs[0],
    [runs, selected],
  );

  function launchRun() {
    if (isRunning) return;
    const id = `Run #${Number(runs[0].id.replace(/\D/g, "")) + 1}`;
    const next: Run = {
      id,
      mode: memoryMode ? "MEMORY" : "COLD",
      time: new Date().toLocaleTimeString([], { hour12: false }),
      status: "running",
      f1: ".710",
      action: "Analyzing…",
      delta: "—",
    };
    setRuns((current) => [next, ...current]);
    setSelected(id);
    setIsRunning(true);
    setActiveStage("data");
    const sequence = ["train", "eval", "memory", "science", "experiment", "critic"];
    sequence.forEach((key, index) => {
      window.setTimeout(() => setActiveStage(key), 650 * (index + 1));
    });
    window.setTimeout(() => {
      setRuns((current) =>
        current.map((run) =>
          run.id === id
            ? { ...run, status: "complete", f1: ".749", action: "Increase epochs", delta: "+.039" }
            : run,
        ),
      );
      setIsRunning(false);
    }, 5000);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /></div>
          <strong>Traceboard</strong>
          <span className="crumb">PCB Intelligence</span>
          <span className="slash">/</span>
          <span className="page-name">Autonomous Lab</span>
        </div>
        <div className="system-meta">
          <span className="status-pill live"><i /> Live</span>
          <span className="status-pill">2,953 samples</span>
          <span className="status-pill">6 classes</span>
          <span className="clock">16:22:37</span>
          <span className="avatar">AI</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="left-panel">
          <div className="panel-scroll">
            <section className="goal-card">
              <div className="section-kicker"><span>AGENT OBJECTIVE</span><button type="button">EDIT</button></div>
              <h1>Improve PCB defect recall without overfitting.</h1>
              <p>Use real measurements and past experiments to choose the next safe intervention.</p>
              <div className="tag-row"><span>Macro F1</span><span>Memory</span><span>Real data</span></div>
            </section>

            <section className="mode-card">
              <div>
                <h2>{memoryMode ? "Experienced agent" : "Cold start"}</h2>
                <p>{memoryMode ? "Atlas lessons shape the next move" : "No historical lessons retrieved"}</p>
              </div>
              <button
                className={`toggle ${memoryMode ? "on" : ""}`}
                onClick={() => setMemoryMode((value) => !value)}
                aria-label="Toggle memory mode"
                aria-pressed={memoryMode}
              ><span /></button>
            </section>

            <button className={`run-button ${isRunning ? "busy" : ""}`} onClick={launchRun} type="button">
              <span>{isRunning ? "◆" : "▶"}</span>{isRunning ? " Agent running" : " Start autonomous run"}
            </button>
            <button className="secondary-button" type="button" onClick={() => setActiveStage("science")}>Inspect next decision</button>

            <div className="section-title"><span className="dot orange" /> DATASET</div>
            <section className="dataset-card">
              <div className="dataset-head"><strong>Kaggle PCB defects</strong><span>READY</span></div>
              <div className="split-row"><b>2,059</b><span>Train</span><b>447</b><span>Validation</span><b>447</b><span>Test</span></div>
              <p>Test set stays sealed until optimization ends.</p>
            </section>

            <div className="section-title"><span className="dot green" /> RECENT ACTIVITY</div>
            <div className="activity-list">
              <div><i className="green" /><span>Lesson stored in Atlas</span><time>now</time></div>
              <div><i className="orange" /><span>OpenRouter reviewed result</span><time>1m</time></div>
              <div><i className="blue" /><span>Validation evaluation complete</span><time>2m</time></div>
            </div>
          </div>
        </aside>

        <section className="canvas-panel">
          <div className="canvas-toolbar">
            <span>AGENT LOOP · 7 STAGES · 1 ACTIVE</span>
            <div className="canvas-actions">
              <button type="button" onClick={() => setZoom((z) => Math.min(1.15, z + 0.05))} aria-label="Zoom in">+</button>
              <button type="button" onClick={() => setZoom((z) => Math.max(0.85, z - 0.05))} aria-label="Zoom out">−</button>
              <button type="button" onClick={() => setZoom(1)} aria-label="Reset view">↻</button>
            </div>
          </div>

          <div className="agent-canvas">
            <div className="flow" style={{ transform: `scale(${zoom})` }}>
              <div className="flow-line top" />
              <div className="flow-line right" />
              <div className="flow-line bottom" />
              {stages.map((stage, index) => (
                <button
                  type="button"
                  key={stage.key}
                  className={`stage-node node-${index + 1} ${stage.tone} ${activeStage === stage.key ? "active" : ""}`}
                  onClick={() => setActiveStage(stage.key)}
                >
                  <span className="node-icon">{["▦", "◎", "⌁", "◇", "✦", "↗", "✓"][index]}</span>
                  <span><small>{stage.eyebrow}</small><strong>{stage.title}</strong><em>{stage.note}</em></span>
                </button>
              ))}
              <div className="loop-label">AUTONOMOUS LEARNING LOOP <span>↻</span></div>
            </div>

            <div className="legend-card">
              <span><i className="cyan" /> Data</span><span><i className="green" /> Memory</span>
              <span><i className="orange" /> Decision</span><span><i className="red" /> Lesson</span>
            </div>

            <section className="decision-strip">
              <div className="decision-icon">✦</div>
              <div><small>ACTIVE STAGE</small><strong>{stages.find((stage) => stage.key === activeStage)?.title}</strong></div>
              <p>{activeStage === "memory" ? "Atlas found five related lessons; two were explicitly cited by the Scientist." : "Inspect this stage to understand how evidence moves through the agent."}</p>
              <span className="confidence">85% confidence</span>
            </section>
          </div>
        </section>

        <aside className="right-panel">
          <div className="runs-section">
            <div className="right-heading"><span><i className="orange" /> EXPERIMENT RUNS</span><button type="button">VIEW ALL</button></div>
            <div className="run-list">
              {runs.slice(0, 5).map((run) => (
                <button type="button" key={run.id} onClick={() => setSelected(run.id)} className={selected === run.id ? "selected" : ""}>
                  <div><strong>{run.id}</strong><span>{run.time} · {run.mode}</span></div>
                  <div className="run-result"><b>{run.f1}</b><span className={`run-status ${run.status}`}>{run.status}</span></div>
                </button>
              ))}
            </div>
          </div>

          <div className="detail-scroll">
            <section className="metric-summary">
              <div className="right-heading"><span><i className="green" /> RUN FINDINGS</span><span className="run-mode">{selectedRun.mode}</span></div>
              <div className="metric-grid">
                <div><small>BASELINE F1</small><strong>.710</strong></div>
                <div><small>BEST F1</small><strong>{selectedRun.f1}</strong></div>
                <div><small>DELTA</small><strong className={selectedRun.delta.startsWith("+") ? "positive" : "negative"}>{selectedRun.delta}</strong></div>
              </div>
            </section>

            <section className="class-performance">
              <div className="mini-heading"><span>PER-CLASS F1</span><span>VALIDATION</span></div>
              {classMetrics.map(([name, value]) => (
                <div className="bar-row" key={name}>
                  <span>{name}</span><div><i style={{ width: `${value * 100}%` }} /></div><b>{value.toFixed(2)}</b>
                </div>
              ))}
            </section>

            <section className="memory-evidence">
              <div className="mini-heading"><span>MEMORY EVIDENCE</span><span>2 USED / 5 FOUND</span></div>
              <article>
                <div className="memory-rank">01</div>
                <div><strong>Class weighting improved a similar recall gap.</strong><p>lesson_person1_weighted_crops</p></div>
                <span>USED</span>
              </article>
              <article>
                <div className="memory-rank">02</div>
                <div><strong>Stronger augmentation previously reduced F1.</strong><p>real_memory_smoke_exp_001_lesson</p></div>
                <span>USED</span>
              </article>
            </section>

            <section className="lesson-card">
              <small>FIREWORKS CRITIC · STORED IN ATLAS</small>
              <strong>Balanced class weights did not improve this failure pattern.</strong>
              <p>Keep the result as negative evidence and avoid repeating it without new supporting signals.</p>
            </section>
          </div>
        </aside>
      </section>
    </main>
  );
}
