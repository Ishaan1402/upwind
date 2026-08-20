import React, { useEffect, useState } from 'react';
import type { CoverageInfo, LocationInfo, ObservationInfo, SignalItem, HypothesisItem } from '../types/aqi';
import { streamWhyExplanation } from '../services/api';
import { X, AlertCircle, Sparkles, CheckCircle2, XCircle, Loader2, Radio, Flame, Wind, Cpu, Search, Gauge, CloudFog, Flag, Users, ChevronDown, ChevronUp, Circle } from 'lucide-react';

interface WhyDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  location: LocationInfo | null;
  observation: ObservationInfo | null;
  coverage?: CoverageInfo | null;
  observationToken?: string | null;
  onSignalsReady?: (data: { signals: SignalItem[]; hypotheses: HypothesisItem[]; open_questions: string[]; map_layers?: any; execution_trace?: any[]; coverage?: CoverageInfo; total_ms?: number }) => void;
}

interface InternalToolStep {
  step: string;
  label: string;
  duration_ms?: number;
  status: 'pending' | 'in_progress' | 'done' | 'absent' | 'warning';
  result?: string;
  icon: any;
}

const DEFAULT_STEPS: InternalToolStep[] = [
  { step: 'weather_vector', label: 'Wind & temperature', status: 'pending', icon: Wind },
  { step: 'aod_density', label: 'Aerosol density (AOD)', status: 'pending', icon: Radio },
  { step: 'hms_scan', label: 'Smoke plume analysis', status: 'pending', icon: CloudFog },
  { step: 'openaq_monitors', label: 'Monitor concentrations', status: 'pending', icon: Gauge },
  { step: 'place_context', label: 'Local population context', status: 'pending', icon: Users },
  { step: 'web_search', label: 'News & incident search', status: 'pending', icon: Search },
  { step: 'wfigs_scan', label: 'Fire incident registry', status: 'pending', icon: Flag },
  { step: 'firms_scan', label: 'Upwind fire hotspots', status: 'pending', icon: Flame },
  { step: 'score_hypotheses', label: 'Scoring hypotheses', status: 'pending', icon: Cpu }
];

export const WhyDrawer: React.FC<WhyDrawerProps> = ({
  isOpen,
  onClose,
  location,
  observation,
  coverage,
  observationToken,
  onSignalsReady: onSignalsReadyProp
}) => {
  const [toolSteps, setToolSteps] = useState<InternalToolStep[]>(DEFAULT_STEPS);
  const [totalWallMs, setTotalWallMs] = useState<number>(0);
  const [isTraceCollapsed, setIsTraceCollapsed] = useState<boolean>(false);
  const [signals, setSignals] = useState<SignalItem[]>([]);
  const [hypotheses, setHypotheses] = useState<HypothesisItem[]>([]);
  const [streamedNarrative, setStreamedNarrative] = useState<string>('');
  const [isStreamingLLM, setIsStreamingLLM] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Drag gesture state for mobile bottom sheet
  const [dragOffsetY, setDragOffsetY] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isClosing, setIsClosing] = useState<boolean>(false);
  const dragStartYRef = React.useRef<number>(0);
  const rafIdRef = React.useRef<number | null>(null);

  const handlePointerDown = (e: React.PointerEvent) => {
    if (e.button !== undefined && e.button !== 0) return;
    // Restrict drag gesture to mobile viewports
    if (typeof window !== 'undefined' && window.innerWidth > 768) return;
    setIsDragging(true);
    setIsClosing(false);
    dragStartYRef.current = e.clientY;
    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch (_) {}
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDragging || isClosing) return;
    const deltaY = e.clientY - dragStartYRef.current;
    const clampedY = Math.max(0, deltaY);
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(() => {
      setDragOffsetY(clampedY);
    });
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    if (!isDragging) return;
    setIsDragging(false);
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch (_) {}

    const finalDeltaY = Math.max(0, e.clientY - dragStartYRef.current);
    if (finalDeltaY > 80) {
      // Animate slide-down exit on mobile before closing
      requestAnimationFrame(() => {
        setIsClosing(true);
        setTimeout(() => {
          onClose();
          setIsClosing(false);
          setDragOffsetY(0);
        }, 240);
      });
    } else {
      setDragOffsetY(0);
    }
  };

  const isGoodAqi = (observation?.aqi ?? 0) <= 50;
  const relevantHypotheses = isGoodAqi
    ? hypotheses.filter(h => h.support.length > 0)
    : hypotheses;

  useEffect(() => {
    setDragOffsetY(0);
    setIsDragging(false);
    setIsClosing(false);

    if (!isOpen || !location || !observation) {
      setToolSteps(DEFAULT_STEPS);
      setIsTraceCollapsed(false);
      setSignals([]);
      setHypotheses([]);
      setStreamedNarrative('');
      setIsStreamingLLM(false);
      setError(null);
      return;
    }

    // Reset state for new SSE stream
    setToolSteps(DEFAULT_STEPS.map(s => ({ ...s, status: 'pending' })));
    setTotalWallMs(0);
    setIsTraceCollapsed(false);
    setSignals([]);
    setHypotheses([]);
    setStreamedNarrative('');
    setIsStreamingLLM(false);
    setError(null);

    const closeStream = streamWhyExplanation(location, observation, {
      onToolStart: (data) => {
        setToolSteps(prev => prev.map(s => s.step === data.step ? { ...s, status: 'in_progress' } : s));
      },
      onToolDone: (data) => {
        const status: InternalToolStep['status'] =
          data.status === 'absent' || data.status === 'warning' ? data.status : 'done';
        setToolSteps(prev => prev.map(s => s.step === data.step ? {
          ...s,
          status,
          duration_ms: data.duration_ms,
          result: data.result
        } : s));
      },
      onSignalsReady: (data) => {
        setSignals(data.signals || []);
        setHypotheses(data.hypotheses || []);
        if (typeof data.total_ms === 'number') setTotalWallMs(data.total_ms);
        onSignalsReadyProp?.(data);
        setIsStreamingLLM(true);
        // Automatically collapse tool execution accordion so narrative takes center stage
        setIsTraceCollapsed(true);
      },
      onToken: (token) => {
        setStreamedNarrative(prev => prev + token);
      },
      onComplete: (data) => {
        setIsStreamingLLM(false);
        if (data.narrative && !streamedNarrative) {
          setStreamedNarrative(data.narrative);
        }
      },
      onError: (err) => {
        setIsStreamingLLM(false);
        setError(err.message || 'Error streaming live evidence explanation.');
      }
    }, observationToken);

    return () => {
      closeStream();
    };
  }, [isOpen, location, observation, observationToken]);

  if (!isOpen) return null;

  // Display Real-time until backend delivers total wall-clock time
  const completedCount = toolSteps.filter(s =>
    s.status === 'done' || s.status === 'absent' || s.status === 'warning'
  ).length;

  // Wind-dependent steps (WFIGS / FIRMS) can't start until the wind vector resolves
  const weatherStep = toolSteps.find(s => s.step === 'weather_vector');
  const weatherResolved = !!weatherStep && (weatherStep.status === 'done' || weatherStep.status === 'warning');
  const isWaitingOnWind = (step: InternalToolStep) =>
    (step.step === 'wfigs_scan' || step.step === 'firms_scan') &&
    step.status === 'pending' && !weatherResolved;

  // Outages are summarized as a muted footnote; only present/absent chips render
  const gridSignals = signals.filter(sig => sig.status !== 'unavailable');
  const unavailableCount = signals.filter(sig => sig.status === 'unavailable').length;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div
        className="drawer-panel"
        onClick={(e) => e.stopPropagation()}
        style={{
          transform: isClosing
            ? 'translate3d(0, 100%, 0)'
            : dragOffsetY > 0
            ? `translate3d(0, ${Math.round(dragOffsetY)}px, 0)`
            : undefined,
          transition: isDragging
            ? 'none'
            : 'transform 0.24s cubic-bezier(0.32, 0.72, 0, 1)',
          willChange: isDragging || isClosing ? 'transform' : 'auto'
        }}
      >
        <div
          className="drawer-drag-zone"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onLostPointerCapture={handlePointerUp}
        >
          <div className="drawer-handle" aria-hidden="true" />
          <div className="drawer-header">
            <div className="flex items-center gap-2">
              <Sparkles className="text-accent" size={20} />
              <h3 className="drawer-title">Air Quality Breakdown</h3>
            </div>
            <button
              onClick={onClose}
              className="drawer-close-btn"
              onPointerDown={(e) => e.stopPropagation()}
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="drawer-body">
          <div className="trace-section">
            {/* Collapsible tool trace bar */}
            {(signals.length > 0 || isTraceCollapsed) && (
              <div
                className="collapsible-trace-bar"
                onClick={() => setIsTraceCollapsed(prev => !prev)}
              >
                <div className="trace-bar-title">
                  <CheckCircle2 size={16} className="text-emerald-400" />
                  <span>{completedCount} Data Sources Checked</span>
                  <span className="trace-bar-badge">{totalWallMs > 0 ? `${(totalWallMs / 1000).toFixed(1)}s` : 'Real-time'}</span>
                </div>
                <div className="flex items-center gap-1 text-zinc-400" style={{ fontSize: '0.75rem' }}>
                  {isTraceCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                </div>
              </div>
            )}

            {/* Tool execution trace container */}
            <div className={`trace-collapse-wrapper ${isTraceCollapsed ? 'is-collapsed' : ''}`}>
              <div className="trace-collapse-inner">
                <div className="tool-execution-container">
                  <div className="tool-steps-list">
                    {toolSteps.map((step, idx) => {
                      const StepIcon = step.icon;
                      const isDone = step.status === 'done';
                      const isInProgress = step.status === 'in_progress';
                      const isPending = step.status === 'pending';
                      const isAbsent = step.status === 'absent';
                      const isWarning = step.status === 'warning';
                      const isResolved = isDone || isAbsent || isWarning;
                      const waiting = isWaitingOnWind(step);
                      // Stagger step animation delays for cascading effect
                      const cascadeDelay = isTraceCollapsed
                        ? (toolSteps.length - 1 - idx) * 18
                        : idx * 18;

                      return (
                        <div
                          key={idx}
                          className={`tool-step-card ${isDone ? 'step-done' : ''} ${isInProgress ? 'step-active' : ''} ${isPending ? 'step-pending' : ''}`}
                          style={{ transitionDelay: `${cascadeDelay}ms` }}
                        >
                          <div className="step-icon-wrapper">
                            {isDone ? (
                              <CheckCircle2 size={16} className="text-emerald-400" />
                            ) : isAbsent ? (
                              <CheckCircle2 size={16} className="text-zinc-500" />
                            ) : isWarning ? (
                              <AlertCircle size={16} className="text-amber-400" />
                            ) : isInProgress ? (
                              <Loader2 size={16} className="animate-spin text-accent" />
                            ) : (
                              <Circle size={16} className="text-zinc-600" />
                            )}
                          </div>

                          <div className="flex flex-col" style={{ flex: 1 }}>
                            <span className="step-label">{step.label}</span>
                            {waiting && (
                              <span className="waiting-on-wind">
                                waiting on wind
                                <span className="waiting-dots" aria-hidden="true">
                                  <span>.</span><span>.</span><span>.</span>
                                </span>
                              </span>
                            )}
                          </div>

                          {isResolved && step.duration_ms !== undefined && (
                            <div className="flex items-center gap-1 text-zinc-400" style={{ fontSize: '0.7rem', fontWeight: 600 }}>
                              <span>{step.duration_ms < 1 ? '<1ms' : `${step.duration_ms}ms`}</span>
                            </div>
                          )}

                          {!isResolved && (
                            <StepIcon size={14} className={isInProgress ? 'text-accent' : 'text-zinc-600'} style={{ marginLeft: 'auto' }} />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {error && (
            <div className="drawer-state-container text-red-400">
              <AlertCircle size={32} />
              <p className="state-text">{error}</p>
            </div>
          )}

          {/* Real-time Streaming LLM Briefing & Evidence Package */}
          {(streamedNarrative || signals.length > 0) && (
            <div className="evidence-container">
              {/* Good AQI Short-Circuit Lead Banner */}
              {isGoodAqi && (
                <div className="good-aqi-lead-banner">
                  <CheckCircle2 size={18} className="text-emerald-400" />
                  <span>Air quality is Good (AQI {observation?.aqi}), no elevated surface pollution to attribute.</span>
                </div>
              )}

              {/* Streaming Narrative Briefing Box */}
              <div className="narrative-box">
                <h4 className="section-subtitle">Briefing</h4>
                {coverage && coverage.mode !== 'us' && (
                  <p className="coverage-note" style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '8px' }}>
                    {coverage.mode === 'international' ? 'International data' : 'Best-effort data'} • {coverage.disclaimer}
                  </p>
                )}
                <p className="narrative-text">
                  {streamedNarrative}
                  {isStreamingLLM && <span className="streaming-cursor">▋</span>}
                </p>
              </div>

              {/* Ranked Hypotheses or Situational Context */}
              {relevantHypotheses.length > 0 && (
                <div className="hypotheses-section">
                  <h4 className="section-subtitle">
                    {isGoodAqi ? "Context" : "Attribution Hypotheses"}
                  </h4>
                  {isGoodAqi && (
                    <p style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '12px' }}>
                      These factors are present nearby in the atmosphere, but surface air quality remains healthy.
                    </p>
                  )}
                  {relevantHypotheses.map((h) => (
                    <div key={h.id} className={`hypothesis-card confidence-${h.confidence}`}>
                      <div className="hypothesis-header">
                        <span className="hypothesis-title">{h.title}</span>
                        <span className={`confidence-tag conf-${h.confidence}`}>
                          <span className="confidence-dot" />
                          {h.confidence} confidence
                        </span>
                      </div>

                      {h.support.length > 0 && (
                        <div className="evidence-list support">
                          <span className="evidence-label">Supports</span>
                          {h.support.map((s, idx) => (
                            <div key={idx} className="evidence-item">
                              <CheckCircle2 size={14} className="text-emerald-400" />
                              <span>{s}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {!isGoodAqi && h.against.length > 0 && (
                        <div className="evidence-list against">
                          <span className="evidence-label">Against</span>
                          {h.against.map((a, idx) => (
                            <div key={idx} className="evidence-item">
                              <XCircle size={14} className="text-amber-400" />
                              <span>{a}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Signals Grid */}
              {signals.length > 0 && (
                <div className="signals-section">
                  <h4 className="section-subtitle">Signals</h4>
                  <div className="signals-grid">
                    {gridSignals.map((sig) => (
                      <div key={sig.id} className={`signal-chip status-${sig.status}`}>
                        <span className="signal-status-dot" />
                        <span className="signal-name">{sig.label}</span>
                        <span className="signal-status-label">{sig.status}</span>
                      </div>
                    ))}
                  </div>
                  {unavailableCount > 0 && (
                    <p className="signals-unavailable-note">
                      {unavailableCount} data source{unavailableCount === 1 ? '' : 's'} unavailable
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  );
};
