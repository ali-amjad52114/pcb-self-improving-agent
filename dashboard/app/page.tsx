"use client";

import { useEffect, useMemo, useState } from "react";
import { apiBase, apiHealthy, getApiRun, listApiRuns, startApiRun, type ApiRun } from "./api-client";

type Run = {
  id: string;
  mode: "MEMORY" | "COLD";
  time: string;
  status: "complete" | "running" | "failed";
  f1: string;
  action: string;
  delta: string;
  raw?: ApiRun;
};

type MemoryItem = { title: string; id: string; used: boolean };

const stageAliases: Record<string, string> = {
  initialize: "data", data: "data", dataset: "data", started: "data",
  train_baseline: "train", training: "train", baseline_trained: "train", experiment_trained: "experiment",
  evaluate_baseline: "eval", evaluating: "eval", baseline_evaluated: "eval", experiment_evaluated: "experiment",
  retrieve_memory: "memory", memory_retrieved: "memory", memory: "memory",
  diagnose: "science", propose_experiment: "science", diagnosis_complete: "science", experiment_proposed: "science",
  run_experiment: "experiment", experiment: "experiment",
  critique: "critic", persist: "critic", complete: "critic", completed: "critic", stopped: "critic",
};

function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function number(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function metric(source: Record<string, unknown>, key: string, fallback = 0): number {
  const direct = source[key];
  if (direct !== undefined) return number(direct, fallback);
  return number(object(source.metrics)[key], fallback);
}

function formatF1(value: number): string {
  return value.toFixed(3).replace(/^0/, "");
}

function normalizeRun(raw: ApiRun): Run {
  const state = object(raw.state ?? raw);
  const current = object(state.current_metrics ?? raw.current_metrics ?? raw.metrics);
  const baseline = object(state.baseline_metrics ?? raw.baseline_metrics);
  const proposal = object(state.proposed_experiment ?? raw.proposal);
  const id = String(raw.run_id ?? raw.id ?? state.run_id ?? "Run");
  const statusValue = String(raw.status ?? state.status ?? "running").toLowerCase();
  const status: Run["status"] = statusValue.includes("fail") || statusValue.includes("error")
    ? "failed" : ["complete", "completed", "stopped", "target_reached", "budget_exhausted"].some((part) => statusValue.includes(part)) ? "complete" : "running";
  const f1 = metric(current, "macro_f1", metric(object(state.best_metrics), "macro_f1"));
  const delta = number(state.last_metric_delta ?? raw.delta, f1 - metric(baseline, "macro_f1", f1));
  const created = String(raw.created_at ?? raw.started_at ?? raw.time ?? "");
  const parsedTime = created ? new Date(created) : null;
  return {
    id,
    mode: String(raw.mode ?? state.mode ?? state.memory_mode ?? "memory").toLowerCase().includes("cold") ? "COLD" : "MEMORY",
    time: parsedTime && !Number.isNaN(parsedTime.valueOf()) ? parsedTime.toLocaleTimeString([], { hour12: false }) : created || "now",
    status,
    f1: formatF1(f1),
    action: String(proposal.next_action ?? raw.action ?? (status === "running" ? "Analyzing…" : "Complete")),
    delta: Math.abs(delta) < 0.0005 ? "—" : `${delta > 0 ? "+" : "−"}${Math.abs(delta).toFixed(3).replace(/^0/, "")}`,
    raw,
  };
}

function stageFor(raw?: ApiRun): string {
  const state = object(raw?.state ?? raw);
  const value = String(raw?.stage ?? raw?.current_stage ?? state.current_stage ?? state.status ?? "").toLowerCase();
  return stageAliases[value] ?? Object.entries(stageAliases).find(([key]) => value.includes(key))?.[1] ?? "data";
}

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
  const [isRunning, setIsRunning] = useState(false);
  const [runningMode, setRunningMode] = useState<"memory" | "cold" | null>(null);
  const [activeStage, setActiveStage] = useState("memory");
  const [zoom, setZoom] = useState(1);
  const [connection, setConnection] = useState<"checking" | "live" | "demo">("checking");

  const selectedRun = useMemo(
    () => runs.find((run) => run.id === selected) ?? runs[0],
    [runs, selected],
  );

  const selectedState = object(selectedRun.raw?.state ?? selectedRun.raw);
  const selectedBaseline = object(selectedState.baseline_metrics);
  const selectedClassMetrics = object(selectedState.per_class_metrics ?? selectedRun.raw?.per_class_metrics);
  const liveClassMetrics = Object.entries(selectedClassMetrics).map(([name, values]) => [
    name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()),
    metric(object(values), "f1"),
  ] as const);
  const displayedClassMetrics = liveClassMetrics.length ? liveClassMetrics : classMetrics;
  const memories = (selectedState.retrieved_lessons ?? selectedRun.raw?.memories ?? []) as unknown[];
  const memoryItems: MemoryItem[] = memories.slice(0, 5).map((value, index) => {
    const item = object(value);
    const usedIds = (object(selectedState.proposed_experiment).memory_used ?? []) as unknown[];
    const id = String(item.memory_id ?? item.lesson_id ?? item.id ?? `memory_${index + 1}`);
    return {
      id,
      title: String(item.lesson ?? item.result ?? item.failure_summary ?? item.summary ?? "Retrieved experiment lesson"),
      used: usedIds.map(String).includes(id) || item.used === true,
    };
  });
  const displayedMemories = memoryItems.length ? memoryItems : [
    { title: "Class weighting improved a similar recall gap.", id: "lesson_person1_weighted_crops", used: true },
    { title: "Stronger augmentation previously reduced F1.", id: "real_memory_smoke_exp_001_lesson", used: true },
  ];
  const dataset = object(selectedState.dataset_summary);
  const critique = object(selectedState.critique);
  const lesson = object(critique.lesson);
  const latestCold = runs.find((run) => run.mode === "COLD" && run.status !== "running");
  const latestMemory = runs.find((run) => run.mode === "MEMORY" && run.status !== "running");
  const comparisonDelta = latestCold && latestMemory
    ? number(latestMemory.f1) - number(latestCold.f1)
    : null;

  useEffect(() => {
    let cancelled = false;
    async function connect() {
      if (!await apiHealthy()) {
        if (!cancelled) setConnection("demo");
        return;
      }
      try {
        const apiRuns = await listApiRuns();
        if (cancelled) return;
        setConnection("live");
        if (apiRuns.length) {
          const normalized = apiRuns.map(normalizeRun);
          setRuns(normalized);
          setSelected(normalized[0].id);
          setIsRunning(normalized.some((run) => run.status === "running"));
          setRunningMode(normalized.find((run) => run.status === "running")?.mode === "COLD" ? "cold" : normalized.some((run) => run.status === "running") ? "memory" : null);
          setActiveStage(stageFor(normalized[0].raw));
        }
      } catch {
        if (!cancelled) setConnection("demo");
      }
    }
    void connect();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (connection !== "live") return;
    const running = runs.find((run) => run.status === "running");
    if (!running) return;
    const timer = window.setInterval(async () => {
      try {
        const updated = normalizeRun(await getApiRun(running.id));
        setRuns((current) => current.map((run) => run.id === running.id ? updated : run));
        setActiveStage(stageFor(updated.raw));
        if (updated.status !== "running") {
          setIsRunning(false);
          setRunningMode(null);
        }
      } catch {
        // A transient poll failure does not discard live data or switch modes.
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [connection, runs]);

  function launchDemoRun(mode: "memory" | "cold") {
    if (isRunning) return;
    const id = `Run #${Number(runs[0].id.replace(/\D/g, "")) + 1}`;
    const next: Run = {
      id,
      mode: mode === "memory" ? "MEMORY" : "COLD",
      time: new Date().toLocaleTimeString([], { hour12: false }),
      status: "running",
      f1: ".710",
      action: "Analyzing…",
      delta: "—",
    };
    setRuns((current) => [next, ...current]);
    setSelected(id);
    setIsRunning(true);
    setRunningMode(mode);
    setActiveStage("data");
    const sequence = ["train", "eval", "memory", "science", "experiment", "critic"];
    sequence.forEach((key, index) => {
      window.setTimeout(() => setActiveStage(key), 650 * (index + 1));
    });
    window.setTimeout(() => {
      setRuns((current) =>
        current.map((run) =>
          run.id === id
            ? mode === "memory"
              ? { ...run, status: "complete", f1: ".749", action: "Increase epochs", delta: "+.039" }
              : { ...run, status: "complete", f1: ".710", action: "Baseline only", delta: "—" }
            : run,
        ),
      );
      setIsRunning(false);
      setRunningMode(null);
    }, 5000);
  }

  async function launchRun(mode: "memory" | "cold") {
    if (isRunning) return;
    if (connection !== "live") {
      launchDemoRun(mode);
      return;
    }
    setIsRunning(true);
    setRunningMode(mode);
    setActiveStage("data");
    try {
      const next = normalizeRun(await startApiRun(mode));
      setRuns((current) => [next, ...current.filter((run) => run.id !== next.id)]);
      setSelected(next.id);
      setActiveStage(stageFor(next.raw));
      if (next.status !== "running") {
        setIsRunning(false);
        setRunningMode(null);
      }
    } catch {
      setConnection("demo");
      setIsRunning(false);
      setRunningMode(null);
      launchDemoRun(mode);
    }
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
          <span className={`status-pill ${connection === "live" ? "live" : "demo"}`} title={connection === "live" ? apiBase : "Backend unavailable; interactive demo data is active"}>
            <i /> {connection === "checking" ? "Checking" : connection === "live" ? "Live API" : "Demo"}
          </span>
          <span className="status-pill">{number(dataset.total_samples, 2953).toLocaleString()} samples</span>
          <span className="status-pill">{Array.isArray(dataset.classes) ? dataset.classes.length : 6} classes</span>
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

            <section className="mode-card compare-intro">
              <div><h2>Run the comparison</h2><p>Same data and model; only Atlas memory changes.</p></div>
            </section>

            <div className="run-choice" aria-label="Choose comparison run">
              <button className={`run-button cold ${runningMode === "cold" ? "busy" : ""}`} onClick={() => launchRun("cold")} disabled={isRunning} type="button">
                <span>{runningMode === "cold" ? "◆" : "○"}</span><b>Run Cold</b><small>No past lessons</small>
              </button>
              <button className={`run-button memory ${runningMode === "memory" ? "busy" : ""}`} onClick={() => launchRun("memory")} disabled={isRunning} type="button">
                <span>{runningMode === "memory" ? "◆" : "◇"}</span><b>Run With Memory</b><small>Atlas-guided</small>
              </button>
            </div>

            <section className="quick-compare" aria-label="Cold versus memory comparison">
              <div className="compare-head"><span>LAST RESULTS</span>{comparisonDelta !== null && <b className={comparisonDelta >= 0 ? "positive" : "negative"}>{comparisonDelta >= 0 ? "+" : ""}{comparisonDelta.toFixed(3)} F1 with memory</b>}</div>
              <div className="compare-columns">
                <button type="button" onClick={() => latestCold && setSelected(latestCold.id)} disabled={!latestCold}>
                  <small>COLD</small><strong>{latestCold?.f1 ?? "—"}</strong><span>0 lessons</span>
                </button>
                <div className="versus">VS</div>
                <button type="button" onClick={() => latestMemory && setSelected(latestMemory.id)} disabled={!latestMemory}>
                  <small>WITH MEMORY</small><strong>{latestMemory?.f1 ?? "—"}</strong><span>{latestMemory ? `${((object(latestMemory.raw?.state ?? latestMemory.raw).retrieved_lessons as unknown[]) ?? []).length || 5} lessons` : "—"}</span>
                </button>
              </div>
            </section>
            <button className="secondary-button" type="button" onClick={() => setActiveStage("science")}>Inspect next decision</button>

            <div className="section-title"><span className="dot orange" /> DATASET</div>
            <section className="dataset-card">
              <div className="dataset-head"><strong>Kaggle PCB defects</strong><span>{connection === "live" ? "LIVE" : "READY"}</span></div>
              <div className="split-row"><b>{number(dataset.train_samples, 2059).toLocaleString()}</b><span>Train</span><b>{number(dataset.validation_samples, 447).toLocaleString()}</b><span>Validation</span><b>{number(dataset.test_samples, 447).toLocaleString()}</b><span>Test</span></div>
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
                  <span><small>{stage.eyebrow}</small><strong>{stage.title}</strong><em>{stage.key === "eval" && selectedRun.raw ? `Macro F1 · ${selectedRun.f1}` : stage.key === "memory" && selectedRun.raw ? `${memories.length} relevant lessons` : stage.key === "science" && selectedRun.raw ? selectedRun.action.replaceAll("_", " ") : stage.key === "experiment" && selectedRun.raw ? `Delta · ${selectedRun.delta}` : stage.note}</em></span>
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
              <p>{activeStage === "memory" ? `Atlas found ${memories.length || 5} related lessons; ${displayedMemories.filter((item) => item.used).length} were explicitly cited by the Scientist.` : "Inspect this stage to understand how evidence moves through the agent."}</p>
              <span className="confidence">{Math.round(number(object(selectedState.proposed_experiment).confidence, .85) * 100)}% confidence</span>
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
                <div><small>BASELINE F1</small><strong>{selectedRun.raw ? formatF1(metric(selectedBaseline, "macro_f1")) : ".710"}</strong></div>
                <div><small>BEST F1</small><strong>{selectedRun.f1}</strong></div>
                <div><small>DELTA</small><strong className={selectedRun.delta.startsWith("+") ? "positive" : "negative"}>{selectedRun.delta}</strong></div>
              </div>
            </section>

            <section className="class-performance">
              <div className="mini-heading"><span>PER-CLASS F1</span><span>VALIDATION</span></div>
              {displayedClassMetrics.map(([name, value]) => (
                <div className="bar-row" key={name}>
                  <span>{name}</span><div><i style={{ width: `${value * 100}%` }} /></div><b>{value.toFixed(2)}</b>
                </div>
              ))}
            </section>

            <section className="memory-evidence">
              <div className="mini-heading"><span>MEMORY EVIDENCE</span><span>{displayedMemories.filter((item) => item.used).length} USED / {displayedMemories.length} FOUND</span></div>
              {displayedMemories.map((item, index) => (
                <article key={item.id}>
                  <div className="memory-rank">{String(index + 1).padStart(2, "0")}</div>
                  <div><strong>{item.title}</strong><p>{item.id}</p></div>
                  <span>{item.used ? "USED" : "FOUND"}</span>
                </article>
              ))}
            </section>

            <section className="lesson-card">
              <small>FIREWORKS CRITIC · STORED IN ATLAS</small>
              <strong>{String(lesson.failure_summary ?? critique.reason ?? "Balanced class weights did not improve this failure pattern.")}</strong>
              <p>{String(lesson.result ?? "Keep the result as negative evidence and avoid repeating it without new supporting signals.")}</p>
            </section>
          </div>
        </aside>
      </section>
    </main>
  );
}
