import React, { useEffect, useState } from 'react';
import type { LocationInfo, ObservationInfo, SignalItem, HypothesisItem } from '../types/aqi';
import { streamWhyExplanation } from '../services/api';
import { X, AlertCircle, Sparkles, CheckCircle2, XCircle, Loader2, Radio, Flame, Wind, Cpu, Search, ChevronDown, ChevronUp, Circle } from 'lucide-react';

// Open Questions is deprecated for now
interface WhyDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  location: LocationInfo | null;
  observation: ObservationInfo | null;
  onSignalsReady?: (data: { signals: SignalItem[]; hypotheses: HypothesisItem[]; open_questions: string[]; map_layers?: any; execution_trace?: any[] }) => void;
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
  { step: 'firms_scan', label: 'Upwind fire hotspots', status: 'pending', icon: Flame },
  { step: 'web_search', label: 'News & incident search', status: 'pending', icon: Search },
  { step: 'score_hypotheses', label: 'Scoring hypotheses', status: 'pending', icon: Cpu }
];

export const WhyDrawer: React.FC<WhyDrawerProps> = ({
  isOpen,
  onClose,
  location,
  observation,
  onSignalsReady: onSignalsReadyProp
}) => {
  const [toolSteps, setToolSteps] = useState<InternalToolStep[]>(DEFAULT_STEPS);
  const [isTraceCollapsed, setIsTraceCollapsed] = useState<boolean>(false);
  const [signals, setSignals] = useState<SignalItem[]>([]);
  const [hypotheses, setHypotheses] = useState<HypothesisItem[]>([]);
  const [streamedNarrative, setStreamedNarrative] = useState<string>('');
  const [isStreamingLLM, setIsStreamingLLM] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Drag-to-dismiss gesture state & refs for mobile bottom sheet
  const [dragOffsetY, setDragOffsetY] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isClosing, setIsClosing] = useState<boolean>(false);
  const dragStartYRef = React.useRef<number>(0);
  const rafIdRef = React.useRef<number | null>(null);

  const handlePointerDown = (e: React.PointerEvent) => {
    if (e.button !== undefined && e.button !== 0) return;
    // Strictly isolate drag gesture to mobile viewports (<= 768px)
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
      // GPU-accelerated slide-down exit animation off the screen before unmounting
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
        setToolSteps(prev => prev.map(s => s.step === data.step ? {
          ...s,
          status: 'done',
          duration_ms: data.duration_ms,
          result: data.result
        } : s));
      },
      onSignalsReady: (data) => {
        setSignals(data.signals || []);
        setHypotheses(data.hypotheses || []);
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
    });

    return () => {
      closeStream();
    };
  }, [isOpen, location, observation]);

  if (!isOpen) return null;

  const totalDuration = toolSteps.reduce((acc, s) => acc + (s.duration_ms || 0), 0);
  const completedCount = toolSteps.filter(s => s.status === 'done').length;

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
            {/* Collapsible Tool Trace Accordion Bar (auto-collapses when signals are ready) */}
            {(signals.length > 0 || isTraceCollapsed) && (
              <div
                className="collapsible-trace-bar"
                onClick={() => setIsTraceCollapsed(prev => !prev)}
              >
                <div className="trace-bar-title">
                  <CheckCircle2 size={16} className="text-emerald-400" />
                  <span>{completedCount} Data Sources Checked</span>
                  <span className="trace-bar-badge">{totalDuration > 0 ? `${(totalDuration / 1000).toFixed(1)}s` : 'Real-time'}</span>
                </div>
                <div className="flex items-center gap-1 text-zinc-400" style={{ fontSize: '0.75rem' }}>
                  {isTraceCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                </div>
              </div>
            )}

            {/* Tool Execution Trace — always mounted so collapse/expand animates
                via CSS (grid-template-rows) instead of an abrupt mount/unmount. */}
            <div className={`trace-collapse-wrapper ${isTraceCollapsed ? 'is-collapsed' : ''}`}>
              <div className="trace-collapse-inner">
                <div className="tool-execution-container">
                  <div className="tool-steps-list">
                    {toolSteps.map((step, idx) => {
                      const StepIcon = step.icon;
                      const isDone = step.status === 'done';
                      const isInProgress = step.status === 'in_progress';
                      const isPending = step.status === 'pending';
                      // Cascade: opening reveals top-to-bottom, closing folds bottom-to-top.
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
                            ) : isInProgress ? (
                              <Loader2 size={16} className="animate-spin text-accent" />
                            ) : (
                              <Circle size={16} className="text-zinc-600" />
                            )}
                          </div>

                          <div className="flex flex-col" style={{ flex: 1 }}>
                            <span className="step-label">{step.label}</span>
                          </div>

                          {isDone && step.duration_ms !== undefined && (
                            <div className="flex items-center gap-1 text-zinc-400" style={{ fontSize: '0.7rem', fontWeight: 600 }}>
                              <span>{step.duration_ms < 1 ? '<1ms' : `${step.duration_ms}ms`}</span>
                            </div>
                          )}

                          {!isDone && (
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
                    {signals.map((sig) => (
                      <div key={sig.id} className={`signal-chip status-${sig.status}`}>
                        <span className="signal-status-dot" />
                        <span className="signal-name">{sig.label}</span>
                        <span className="signal-status-label">{sig.status}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
