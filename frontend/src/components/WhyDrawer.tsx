import React, { useEffect, useState } from 'react';
import type { LocationInfo, ObservationInfo, SignalItem, HypothesisItem } from '../types/aqi';
import { streamWhyExplanation } from '../services/api';
import { X, AlertCircle, Sparkles, CheckCircle2, XCircle, Loader2, Radio, Flame, Wind, Cpu, Search, ChevronDown, ChevronUp, Circle } from 'lucide-react';

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
  const [openQuestions, setOpenQuestions] = useState<string[]>([]);
  const [streamedNarrative, setStreamedNarrative] = useState<string>('');
  const [isStreamingLLM, setIsStreamingLLM] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !location || !observation) {
      setToolSteps(DEFAULT_STEPS);
      setIsTraceCollapsed(false);
      setSignals([]);
      setHypotheses([]);
      setOpenQuestions([]);
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
    setOpenQuestions([]);
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
        setOpenQuestions(data.open_questions || []);
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
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div className="flex items-center gap-2">
            <Sparkles className="text-accent" size={20} />
            <h3 className="drawer-title">Debrief</h3>
          </div>
          <button onClick={onClose} className="drawer-close-btn">
            <X size={20} />
          </button>
        </div>

        <div className="drawer-body">
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

          {/* Expanded Tool Execution Trace */}
          {(!isTraceCollapsed || signals.length === 0) && (
            <div className="tool-execution-container" style={{ marginBottom: '16px' }}>
              <div className="tool-steps-list">
                {toolSteps.map((step, idx) => {
                  const StepIcon = step.icon;
                  const isDone = step.status === 'done';
                  const isInProgress = step.status === 'in_progress';
                  const isPending = step.status === 'pending';

                  return (
                    <div
                      key={idx}
                      className={`tool-step-card ${isDone ? 'step-done' : ''} ${isInProgress ? 'step-active' : ''} ${isPending ? 'step-pending' : ''}`}
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
          )}

          {error && (
            <div className="drawer-state-container text-red-400">
              <AlertCircle size={32} />
              <p className="state-text">{error}</p>
            </div>
          )}

          {/* Real-time Streaming LLM Briefing & Evidence Package */}
          {(streamedNarrative || signals.length > 0) && (
            <div className="evidence-container">
              {/* Streaming Narrative Briefing Box */}
              <div className="narrative-box">
                <h4 className="section-subtitle">Evidence Briefing</h4>
                <p className="narrative-text">
                  {streamedNarrative}
                  {isStreamingLLM && <span className="streaming-cursor">▋</span>}
                </p>
              </div>

              {/* Ranked Hypotheses */}
              {hypotheses.length > 0 && (
                <div className="hypotheses-section">
                  <h4 className="section-subtitle">Hypotheses</h4>
                  {hypotheses.map((h) => (
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

                      {h.against.length > 0 && (
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

              {/* Open Questions */}
              {openQuestions.length > 0 && (
                <div className="hypotheses-section">
                  <h4 className="section-subtitle">Open Questions</h4>
                  <div className="flex flex-col gap-2">
                    {openQuestions.map((q, idx) => (
                      <div key={idx} className="hypothesis-card" style={{ borderColor: 'rgba(251, 191, 36, 0.3)', backgroundColor: 'rgba(251, 191, 36, 0.05)' }}>
                        <div className="flex items-start gap-2 text-amber-300" style={{ fontSize: '0.85rem' }}>
                          <AlertCircle size={16} className="mt-0.5 shrink-0" />
                          <span>{q}</span>
                        </div>
                      </div>
                    ))}
                  </div>
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

        <div style={{ padding: '12px 20px', borderTop: '1px solid var(--border-color)', fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
          Ranked observational hypotheses, not chemical source apportionment.
        </div>
      </div>
    </div>
  );
};
