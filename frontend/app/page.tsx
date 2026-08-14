"use client";

import { useEffect, useRef, useState } from "react";

type Phase = "idle" | "running" | "memory" | "transfer" | "done";

const demoDatasets = [
  { id: "deep-pcb", name: "DeepPCB", detail: "1,500 image pairs · 6 defect classes", tag: "SEVERE IMBALANCE" },
  { id: "pcb-aoi", name: "Factory AOI — Line 4", detail: "840 boards · simulated shift data", tag: "DOMAIN SHIFT" },
];

const soundscapes = [
  { id: "signal", name: "Signal Drift", note: "Focused · minimal", notes: [110, 165, 220], wave: "sine" as OscillatorType },
  { id: "memory", name: "Memory Bloom", note: "Warm · reflective", notes: [131, 196, 262], wave: "sine" as OscillatorType },
  { id: "circuit", name: "Circuit Rain", note: "Digital · precise", notes: [147, 220, 294], wave: "triangle" as OscillatorType },
  { id: "factory", name: "Night Factory", note: "Low · industrial", notes: [82, 123, 164], wave: "sawtooth" as OscillatorType },
  { id: "quiet", name: "Quiet Lab", note: "Airy · calm", notes: [174, 261, 349], wave: "sine" as OscillatorType },
];

const coldExperiments = [
  {
    model: "Baseline CNN",
    meta: "224px · uniform sampling",
    score: 55,
    metric: "Macro F1",
    tone: "#111111",
    defects: ["Missing hole 24%", "Open circuit 31%", "Short 71%"],
    critique: "Minority defects are being ignored. Missing-hole recall is critically low.",
    move: "Apply weighted sampling before changing architecture.",
  },
  {
    model: "Weighted sampling",
    meta: "inverse-frequency sampler",
    score: 61,
    metric: "Macro F1",
    tone: "#111111",
    defects: ["Missing hole 46%", "Open circuit 38%", "Short 70%"],
    critique: "Minority recall recovered. Small open circuits remain difficult to resolve.",
    move: "Tighten crops around candidate defect regions.",
  },
  {
    model: "Localized crops",
    meta: "defect ROI · 1.35× zoom",
    score: 69,
    metric: "Macro F1",
    tone: "#111111",
    defects: ["Missing hole 58%", "Open circuit 55%", "Short 72%"],
    critique: "Localized crops expose fine defects, but resolution limits thin-trace features.",
    move: "Increase input resolution and preserve small features.",
  },
  {
    model: "High-res training",
    meta: "384px · edge-safe augmentation",
    score: 78,
    metric: "Macro F1",
    tone: "#111111",
    defects: ["Missing hole 72%", "Open circuit 69%", "Short 81%"],
    critique: "Fine defects are now visible. Confidence thresholds still suppress true positives.",
    move: "Calibrate per-class confidence thresholds.",
  },
  {
    model: "Calibrated classifier",
    meta: "per-class thresholds · v5",
    score: 84,
    metric: "Macro F1",
    tone: "#111111",
    defects: ["Missing hole 82%", "Open circuit 80%", "Short 88%"],
    critique: "Target reached. The hypothesis chain produced balanced defect recall.",
    move: "Store the verified strategy as reusable experience.",
  },
];

const transferExperiments = [
  { model: "Memory-seeded CNN", meta: "weighted · 384px · ROI crops", score: 74, metric: "Macro F1", tone: "#111111", defects: ["Missing hole 70%", "Open circuit 66%", "Short 82%"] },
  { model: "Calibrated classifier", meta: "thresholds transferred + tuned", score: 84, metric: "Macro F1", tone: "#111111", defects: ["Missing hole 81%", "Open circuit 80%", "Short 88%"] },
];

function useSound(enabled: boolean) {
  const context = useRef<AudioContext | null>(null);
  const play = (kind: "tap" | "step" | "memory" | "win") => {
    if (!enabled || typeof window === "undefined") return;
    const AudioCtx = window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    context.current ||= new AudioCtx();
    const ctx = context.current;
    const now = ctx.currentTime;
    const notes = kind === "win" ? [523, 659, 784] : kind === "memory" ? [330, 494] : kind === "step" ? [220] : [160];
    notes.forEach((frequency, index) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = kind === "memory" ? "sine" : "triangle";
      osc.frequency.setValueAtTime(frequency, now + index * 0.08);
      gain.gain.setValueAtTime(0.0001, now + index * 0.08);
      gain.gain.exponentialRampToValueAtTime(0.08, now + index * 0.08 + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.08 + 0.18);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + index * 0.08);
      osc.stop(now + index * 0.08 + 0.2);
    });
  };
  return play;
}

function useAmbient(enabled: boolean, soundscape: typeof soundscapes[number]) {
  const context = useRef<AudioContext | null>(null);
  useEffect(() => {
    if (!enabled || typeof window === "undefined") return;
    const AudioCtx = window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    context.current ||= new AudioCtx();
    const ctx = context.current;
    void ctx.resume();
    let step = 0;
    const playTone = () => {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();
      osc.type = soundscape.wave;
      osc.frequency.value = soundscape.notes[step % soundscape.notes.length];
      filter.type = "lowpass";
      filter.frequency.value = soundscape.id === "circuit" ? 1100 : 620;
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(soundscape.id === "factory" ? 0.018 : 0.026, now + 0.3);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 2.2);
      osc.connect(filter).connect(gain).connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 2.3);
      step += 1;
    };
    playTone();
    const timer = window.setInterval(playTone, soundscape.id === "circuit" ? 1050 : 1600);
    return () => window.clearInterval(timer);
  }, [enabled, soundscape]);
}

export default function Home() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [visible, setVisible] = useState(0);
  const [guided, setGuided] = useState(false);
  const [soundOn, setSoundOn] = useState(false);
  const [soundMenu, setSoundMenu] = useState(false);
  const [soundscape, setSoundscape] = useState(soundscapes[0]);
  const [clock, setClock] = useState("");
  const [dataset, setDataset] = useState(demoDatasets[0]);
  const [fileName, setFileName] = useState("");
  const play = useSound(soundOn);
  useAmbient(soundOn, soundscape);

  useEffect(() => {
    const update = () => setClock(new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date()));
    update();
    const timer = window.setInterval(update, 30000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (phase !== "running" && phase !== "transfer") return;
    const total = phase === "running" ? coldExperiments.length : transferExperiments.length;
    if (visible >= total) {
      const timer = window.setTimeout(() => {
        if (phase === "running") {
          play("memory");
          setPhase("memory");
        } else {
          play("win");
          setPhase("done");
        }
      }, 700);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => {
      play("step");
      setVisible((value) => value + 1);
    }, visible === 0 ? 400 : 1050);
    return () => window.clearTimeout(timer);
  }, [phase, visible, play]);

  useEffect(() => {
    if (!guided || phase !== "memory") return;
    const timer = window.setTimeout(() => {
      play("memory");
      setVisible(0);
      setPhase("transfer");
    }, 1800);
    return () => window.clearTimeout(timer);
  }, [guided, phase, play]);

  const startCold = () => {
    setGuided(false);
    play("tap");
    setVisible(0);
    setPhase("running");
  };

  const chooseDataset = (next: typeof demoDatasets[number]) => {
    setDataset(next);
    setFileName("");
    play("tap");
  };

  const startTransfer = () => {
    setGuided(false);
    play("memory");
    setVisible(0);
    setPhase("transfer");
  };

  const startDemoFlow = () => {
    setGuided(true);
    setDataset(demoDatasets[0]);
    setFileName("");
    setVisible(0);
    setPhase("running");
    play("tap");
    window.requestAnimationFrame(() => document.getElementById("demo")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };

  const active = phase === "transfer" ? transferExperiments : coldExperiments;
  const flowNarration = phase === "running"
    ? visible === 0
      ? "Profiling the PCB dataset and measuring the 55% macro F1 baseline."
      : `Fireworks is critiquing experiment ${Math.min(visible, coldExperiments.length)} and choosing the next real training intervention.`
    : phase === "memory"
      ? "MongoDB is saving the verified failure pattern, intervention, outcome and reusable lesson."
      : phase === "transfer"
        ? "Atlas Vector Search and Voyage AI found a similar failure, so the agent starts with experience instead of guessing."
        : phase === "done"
          ? "OpenRouter verifies the final lesson: the experienced agent reached the same target with 60% fewer experiments."
          : "Ready to run the complete cold-start, memory and transfer-learning story.";

  return (
    <main className="monochrome">
      <nav>
        <a className="brand" href="#top" aria-label="Switchback AI home">
          <span className="brand-mark">S</span>
          <span>SWITCHBACK AI</span>
        </a>
        <div className="nav-center nav-links"><a href="#proof">Why it wins</a><a href="#demo">Live demo</a><a href="#technology">Technology</a></div>
        <span className="mock-badge"><i /> DEMO READY</span>
      </nav>

      <div className="context-bar"><span>SAN FRANCISCO, CA</span><i /> <time>{clock || "LOCAL TIME"}</time><i /> <span>PCB AGENT DEMO</span></div>

      <section className="hero" id="top">
        <div className="eyebrow"><span>NO COLD START</span><i />Persistent intelligence for AutoML</div>
        <h1>Inspect smarter.<br /><em>Learn every time.</em></h1>
        <p className="lede">A PCB inspection agent that learns how to improve classifiers—not just one classifier. Persistent experience cuts the path to the same target from five experiments to two.</p>
        <div className="hero-actions">
          <button className="primary hero-link" onClick={startDemoFlow}><span>DEMO FLOW</span><b>▶</b></button>
          <div className="mini-proof"><strong>5 → 2</strong><span>experiments to target</span></div>
        </div>
      </section>

      <section className="judge-brief" id="proof" aria-label="Project summary">
        <div className="brief-intro">
          <span className="section-index">THE TEN-SECOND STORY</span>
          <h2>From trial and error<br />to transferable experience.</h2>
        </div>
        <div className="brief-grid">
          <article><small>THE PROBLEM</small><strong>ML tuning starts from zero</strong><p>Teams repeat expensive experiments when a new but similar dataset arrives.</p></article>
          <article><small>THE AGENT</small><strong>Diagnoses and intervenes</strong><p>It reads failure evidence, chooses a real training action, and measures the result.</p></article>
          <article><small>THE MEMORY</small><strong>MongoDB preserves why</strong><p>Problem fingerprints, actions, outcomes and verified lessons become reusable experience.</p></article>
          <article className="brief-result"><small>THE PROOF</small><strong>5 → 2</strong><p>Sixty percent fewer experiments to reach the same 84% macro F1 target.</p></article>
        </div>
      </section>

      <section className="demo-shell" id="demo">
        <div className="demo-toolbar">
          <div><span className="window-dot" /><span className="window-dot" /><span className="window-dot" /></div>
          <strong>SWITCHBACK AI / PCB AGENT</strong>
          <span>{guided && phase !== "done" ? "DEMO FLOW RUNNING" : "GUIDED MOCK · API READY"}</span>
        </div>
        <div className="demo-sidebar">
          <span className="sidebar-label">DATA SOURCE</span>
          {demoDatasets.map((item) => (
            <button key={item.id} className={dataset.id === item.id && !fileName ? "dataset-option selected" : "dataset-option"} onClick={() => chooseDataset(item)}>
              <span className="dataset-icon">▦</span><span><strong>{item.name}</strong><small>{item.detail}</small></span>
            </button>
          ))}
          <label className={fileName ? "upload-option selected" : "upload-option"}>
            <input type="file" accept=".zip,.csv,.json" onChange={(event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              setFileName(file.name);
              setDataset({ id: "upload", name: file.name, detail: "Local mock dataset", tag: "UNPROFILED" });
              play("tap");
            }} />
            <span>＋</span><strong>{fileName || "Upload dataset"}</strong><small>ZIP, CSV or JSON</small>
          </label>
          <div className="agent-stack-mini">
            <span className="sidebar-label">AGENT SERVICES</span>
            <p><i className={phase !== "idle" ? "online" : ""} /> MongoDB Atlas <b>MEMORY</b></p>
            <p><i className={phase === "running" || phase === "transfer" ? "online" : ""} /> Fireworks AI <b>REASONING</b></p>
            <p><i className={phase === "memory" || phase === "transfer" || phase === "done" ? "online" : ""} /> Voyage AI <b>RETRIEVAL</b></p>
            <p><i className={phase === "done" ? "online" : ""} /> OpenRouter <b>VERIFY</b></p>
          </div>
        </div>
        <section className="lab" id="learning-loop" aria-live="polite">
        <header className="lab-header">
          <div>
            <span className="section-index">01 / THE LEARNING LOOP</span>
            <h2>{phase === "transfer" || phase === "done" ? "PCB batch B · transfer run" : "PCB batch A · cold run"}</h2>
          </div>
          <div className="dataset-pill">
            <span className="pulse-icon">⌁</span>
            <div><small>{dataset.tag}</small><strong>{dataset.name} · {dataset.detail}</strong></div>
          </div>
        </header>

        <div className="loop-strip">
          {["PROFILE", "RECALL", "PROPOSE", "TRAIN", "CRITIQUE", "LEARN"].map((item, index) => (
            <div className={(phase !== "idle" && index <= Math.min(5, visible + 1)) ? "loop-node active" : "loop-node"} key={item}>
              <span>{String(index + 1).padStart(2, "0")}</span>{item}
            </div>
          ))}
        </div>

        {guided && phase !== "idle" && (
          <>
            <div className="guided-flow-bar" aria-label="Demo flow progress">
              <b>{phase === "done" ? "DEMO FLOW COMPLETE" : "DEMO FLOW"}</b>
              <span className={phase === "running" ? "active" : "complete"}>1 · COLD RUN</span><i>→</i>
              <span className={phase === "memory" ? "active" : phase === "transfer" || phase === "done" ? "complete" : ""}>2 · STORE MEMORY</span><i>→</i>
              <span className={phase === "transfer" ? "active" : phase === "done" ? "complete" : ""}>3 · TRANSFER</span><i>→</i>
              <span className={phase === "done" ? "active" : ""}>4 · PROOF</span>
            </div>
            <div className="flow-narrator" key={`${phase}-${visible}`}><span>NOW HAPPENING</span><p>{flowNarration}</p><b>{phase === "done" ? "VERIFIED" : "LIVE"}</b></div>
          </>
        )}

        {phase === "idle" ? (
          <div className="launch-panel">
            <div className="orb"><div className="orb-core">55</div><span>BASELINE F1</span></div>
            <div>
              <span className="kicker">NEW PROBLEM DETECTED</span>
              <h3>Can an inspector<br />improve its own training?</h3>
              <p>Start with weak minority-defect recall. The agent will diagnose evidence, execute real interventions, and remember every outcome.</p>
              <button className="text-button" onClick={startCold}>RUN AUTONOMOUS AGENT <span>→</span></button>
            </div>
          </div>
        ) : phase === "memory" ? (
          <MemoryCard onTransfer={startTransfer} guided={guided} />
        ) : phase === "done" ? (
          <ResultCard onReset={startDemoFlow} />
        ) : (
          <div className="experiments">
            {phase === "transfer" && (
              <div className="memory-hit">
                <div className="radar"><i /><i /><i /></div>
                <div><span>ATLAS VECTOR SEARCH · 94% MATCH</span><strong>Relevant PCB failure found</strong><p>Small-defect recall + class imbalance → weighted sampling, ROI crops, and higher resolution.</p></div>
                <b>MEMORY APPLIED</b>
              </div>
            )}
            {active.slice(0, visible).map((experiment, index) => (
              <article className="experiment-card" style={{ "--tone": experiment.tone } as React.CSSProperties} key={`${phase}-${index}`}>
                <div className="exp-number">{String(index + 1).padStart(2, "0")}</div>
                <div className="exp-model"><small>EXPERIMENT</small><h3>{experiment.model}</h3><code>{experiment.meta}</code></div>
                <div className="score-ring" style={{ "--score": `${experiment.score * 3.6}deg` } as React.CSSProperties}>
                  <div><strong>{experiment.score}%</strong><span>{experiment.metric}</span></div>
                </div>
                {"critique" in experiment ? (
                  <div className="critic"><small>FIREWORKS CRITIC</small><p>“{experiment.critique}”</p><span>NEXT MOVE · {experiment.move}</span></div>
                ) : (
                  <div className="critic compact"><small>MEMORY-SEEDED SEARCH</small><p>{index === 0 ? "Skipped three known-weak experiments." : "Transferred the winning calibration strategy."}</p><span>{index + 1} / 2 TO TARGET</span></div>
                )}
                <div className="defect-strip" aria-label="Per-class recall evidence">
                  {experiment.defects.map((defect) => <span key={defect}>{defect}</span>)}
                </div>
              </article>
            ))}
            <div className="agent-thinking"><span /><span /><span /> {visible < active.length ? "Agent is choosing the next experiment" : "Distilling outcome into experience"}</div>
          </div>
        )}
        </section>
      </section>

      <p className="demo-disclosure"><b>TRANSPARENT DEMO MODE</b> The interface currently replays representative agent events. When connected, the same components consume live backend run, experiment, memory and verification events.</p>

      <section className="technical-proof" id="technology">
        <header>
          <span className="section-index">02 / THE TECHNICAL PROOF</span>
          <h2>The model improves.<br /><em>The experimenter learns.</em></h2>
          <p>A normal training run produces a model. Switchback AI also produces a reusable lesson, grounded in what changed and whether it worked.</p>
        </header>
        <div className="proof-cards">
          <article><small>OBJECTIVE EVIDENCE</small><strong>55 → 84</strong><span>Macro F1 across the cold run</span></article>
          <article><small>PERSISTENT LEARNING</small><strong>1 verified lesson</strong><span>Stored with state, intervention and outcome</span></article>
          <article><small>TRANSFER EFFICIENCY</small><strong>60% fewer trials</strong><span>Same target on a related PCB batch</span></article>
        </div>
        <div className="evidence-ledger" aria-label="Technology evidence ledger">
          {[
            ["MongoDB Atlas", "System of record", "Run state, experiments, metrics and causal lessons", "MEMORY RETRIEVED"],
            ["Atlas Vector Search + Voyage AI", "Experience retrieval", "Embeds failure signatures and finds relevant past cases", "94% MATCH"],
            ["Fireworks AI", "Actor + critic", "Proposes interventions and diagnoses why a run failed", "NEXT MOVE CHOSEN"],
            ["OpenRouter", "Independent verification", "Challenges low-confidence or final distilled lessons", "LESSON VERIFIED"],
            ["LangGraph + LangSmith", "Control + observability", "Orchestrates the loop and traces every decision", "AUDITABLE RUN"],
          ].map(([name, role, detail, output]) => (
            <div key={name}><strong>{name}</strong><span>{role}</span><p>{detail}</p><b>{output}</b></div>
          ))}
        </div>
      </section>

      <section className="manifesto">
        <span className="section-index">03 / REAL-WORLD IMPACT</span>
        <blockquote>“Fewer blind experiments.<br />Faster quality systems.<br /><em>Less compute wasted.</em>”</blockquote>
        <div className="responsibility-grid">
          <div><small>HUMAN CONTROL</small><strong>The agent recommends training changes; people approve deployment.</strong></div>
          <div><small>HONEST METRICS</small><strong>Per-class recall exposes minority defects that accuracy can hide.</strong></div>
          <div><small>AUDITABILITY</small><strong>Every hypothesis, action, outcome and lesson remains traceable.</strong></div>
        </div>
      </section>

      <footer><span>SWITCHBACK AI / 2026</span><p>Every experiment makes the next one smarter.</p><span>BUILT AT MONGODB .LOCAL</span></footer>
      {soundMenu && (
        <div className="sound-menu" role="dialog" aria-label="Choose ambient soundscape">
          <div className="sound-menu-head"><div><small>AMBIENT AUDIO</small><strong>{soundOn ? "Playing" : "Sound is off"}</strong></div><button onClick={() => setSoundOn((value) => !value)}>{soundOn ? "PAUSE" : "PLAY"}</button></div>
          {soundscapes.map((item, index) => (
            <button className={soundscape.id === item.id ? "sound-choice selected" : "sound-choice"} key={item.id} onClick={() => { setSoundscape(item); setSoundOn(true); }}>
              <span>{String(index + 1).padStart(2, "0")}</span><div><strong>{item.name}</strong><small>{item.note}</small></div><i>{soundscape.id === item.id && soundOn ? "◼︎" : "▶"}</i>
            </button>
          ))}
          <p>Original generative tones · no copyrighted audio</p>
        </div>
      )}
      <button className="sound sound-float" onClick={() => setSoundMenu((value) => !value)} aria-label="Choose ambient soundscape" aria-expanded={soundMenu} title="Ambient soundscapes">
        <span aria-hidden="true">{soundOn ? "◖))" : "◖×"}</span>
      </button>
    </main>
  );
}

function MemoryCard({ onTransfer, guided }: { onTransfer: () => void; guided: boolean }) {
  return (
    <div className="memory-card">
      <div className="memory-visual"><div className="memory-cube">◆</div><span>SAVED TO MONGODB</span></div>
      <div className="memory-copy">
        <span className="kicker">EXPERIENCE DISTILLED</span>
        <h3>The agent now knows<br />what worked—and why.</h3>
        <div className="lesson-grid">
          <div><small>FAILURE</small><strong>Low recall on small minority defects</strong></div>
          <div><small>FAILED START</small><strong>Uniform 224px training</strong></div>
          <div><small>WINNING STRATEGY</small><strong>Weighting + ROI crops + 384px</strong></div>
          <div><small>OBSERVED LIFT</small><strong className="green">+29 macro F1</strong></div>
        </div>
        <button className="primary cyan" onClick={onTransfer} disabled={guided}><span>{guided ? "APPLYING MEMORY…" : "TEST TRANSFER LEARNING"}</span><b>→</b></button>
      </div>
    </div>
  );
}

function ResultCard({ onReset }: { onReset: () => void }) {
  return (
    <div className="result-card">
      <span className="confetti c1">✦</span><span className="confetti c2">●</span><span className="confetti c3">◆</span><span className="confetti c4">✦</span>
      <span className="kicker">TRANSFER COMPLETE</span>
      <h3>Same 84% target.<br /><em>Reached in two experiments.</em></h3>
      <div className="race">
        <div><small>COLD AGENT</small><strong>5</strong><span>experiments to 84% F1</span><i style={{ width: "100%" }} /></div>
        <div className="winner"><small>EXPERIENCED AGENT</small><strong>2</strong><span>experiments to 84% F1</span><i style={{ width: "40%" }} /></div>
      </div>
      <p>MongoDB memory changed the agent’s starting strategy. The model didn’t just improve—the experimenter did.</p>
      <button className="text-button" onClick={onReset}>REPLAY DEMO FLOW <span>↻</span></button>
    </div>
  );
}
