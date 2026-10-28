import { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  Activity, 
  Lock, 
  Cpu, 
  Eye, 
  FileCheck2, 
  AlertCircle,
  Hash,
  Sparkles
} from 'lucide-react';
import { fetchHealth, verifyAuditLedger, fetchAuditEntries } from './services/api';
import { HealthResponse, AuditEntry, AuditVerificationResult, FaceSpec } from './types';

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [auditResult, setAuditResult] = useState<AuditVerificationResult | null>(null);
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [showAuditModal, setShowAuditModal] = useState(false);

  // Default initial FaceSpec: All features unknown per Rule 3
  const [faceSpec] = useState<FaceSpec>({
    gender_presentation: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    approximate_age: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    face_shape: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    skin_tone_complexion: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    hair_style_color: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    eyebrows: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    eyes_shape_color: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    nose_shape: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    mouth_lips: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    jawline_chin: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    facial_hair: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    distinguishing_marks: { value: "unknown", confidence: 0.0, stated: false, locked: false },
    accessories: { value: "unknown", confidence: 0.0, stated: false, locked: false }
  });

  useEffect(() => {
    async function loadData() {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch (e) {
        console.warn("Backend not yet reachable on localhost:8000:", e);
      } finally {
        setHealthLoading(false);
      }
    }
    loadData();
  }, []);

  const handleVerifyLedger = async () => {
    try {
      const res = await verifyAuditLedger();
      setAuditResult(res);
      const entries = await fetchAuditEntries(20);
      setAuditEntries(entries);
      setShowAuditModal(true);
    } catch (e) {
      alert(`Could not verify audit chain: ${e}`);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-3.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Cpu className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-semibold tracking-tight text-white text-base">Forensic Face Composite Studio</h1>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">PS-01</span>
            </div>
            <p className="text-xs text-slate-400">Cognitive Elicitation & Evidentiary Composite System</p>
          </div>
        </div>

        {/* System & Model Status Bar */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <Activity className="h-3.5 w-3.5 text-slate-400" />
            <span className="text-slate-400">API:</span>
            {healthLoading ? (
              <span className="text-amber-400 animate-pulse">Connecting...</span>
            ) : health?.status === 'ok' ? (
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span> Online
              </span>
            ) : (
              <span className="text-amber-400">Degraded / Local Mock</span>
            )}
          </div>

          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-800">
            <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
            <span className="text-slate-400">Text:</span>
            <span className="text-cyan-300 font-medium">{health?.gemini_text_model || "gemini-2.5-flash"}</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">Image:</span>
            <span className="text-cyan-300 font-medium">{health?.gemini_image_model || "imagen-3.0-generate-002"}</span>
          </div>

          <button
            onClick={handleVerifyLedger}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-cyan-950/60 hover:bg-cyan-900/80 text-cyan-300 border border-cyan-800/60 transition cursor-pointer"
          >
            <ShieldCheck className="h-3.5 w-3.5 text-cyan-400" />
            <span>Audit Ledger</span>
          </button>
        </div>
      </header>

      {/* Main Studio Viewport */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* Left Column: Cognitive Interview Dialogue (4 cols) */}
        <section className="lg:col-span-4 border-r border-slate-800 flex flex-col bg-slate-900/40">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
            <div className="flex items-center gap-2">
              <Eye className="h-4 w-4 text-cyan-400" />
              <h2 className="font-semibold text-sm">Cognitive Elicitation Console</h2>
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800 text-emerald-400">
              Anti-Leading Active
            </span>
          </div>

          <div className="p-3 bg-slate-950/40 border-b border-slate-800/80 text-xs text-slate-400 flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
            <span>
              <strong>Rule 7:</strong> Investigator agent strictly avoids leading questions. Focus on free recall and unprompted morphological descriptions.
            </span>
          </div>

          <div className="flex-1 p-4 overflow-y-auto space-y-3 font-sans text-sm">
            <div className="p-3.5 rounded-lg bg-slate-800/60 border border-slate-700/60 text-slate-200">
              <div className="text-[11px] font-mono text-cyan-400 mb-1 flex items-center gap-1.5">
                <FileCheck2 className="h-3 w-3" /> Certified Forensic Agent
              </div>
              "Hello. Take all the time you need. Close your eyes if that helps you picture the scene. In your own words, describe whatever facial details stand out most clearly in your memory."
            </div>
          </div>

          <div className="p-4 border-t border-slate-800 bg-slate-900/80">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Witness statement (e.g. 'He had prominent cheekbones and dark hair')..."
                className="flex-1 bg-slate-950 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
              />
              <button className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium rounded-md transition cursor-pointer">
                Submit
              </button>
            </div>
          </div>
        </section>

        {/* Center Column: Composite Canvas Viewport (5 cols) */}
        <section className="lg:col-span-5 flex flex-col bg-slate-950 border-r border-slate-800">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/40">
            <div className="flex items-center gap-2">
              <h2 className="font-semibold text-sm">Forensic Composite Canvas</h2>
              <span className="text-xs font-mono text-slate-500">v1.0 (Baseline)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-amber-400 bg-amber-950/40 border border-amber-800/60 px-2 py-0.5 rounded flex items-center gap-1">
                <Lock className="h-3 w-3" /> Region Lock Enforced
              </span>
            </div>
          </div>

          <div className="flex-1 p-6 flex flex-col items-center justify-center relative">
            {/* Canvas Frame with Mandatory Statutory Watermark per Rule 6 */}
            <div className="w-full max-w-md aspect-square bg-slate-900 rounded-lg border border-slate-800 overflow-hidden shadow-2xl relative flex flex-col items-center justify-center">
              {/* Mandatory Top Watermark Banner */}
              <div className="absolute top-0 inset-x-0 bg-red-950/80 backdrop-blur-sm border-b border-red-800/60 py-1 px-3 text-center z-10">
                <p className="text-[10px] font-mono uppercase tracking-wider text-red-200 font-semibold">
                  AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY
                </p>
              </div>

              {/* Placeholder forensic sketch rendering */}
              <div className="text-center p-8 space-y-3">
                <div className="h-32 w-32 mx-auto rounded-full border-2 border-dashed border-slate-700 flex items-center justify-center text-slate-600">
                  <Eye className="h-12 w-12 stroke-[1.2]" />
                </div>
                <div>
                  <h3 className="font-medium text-slate-300 text-sm">Awaiting Witness Initial Description</h3>
                  <p className="text-xs text-slate-500 max-w-xs mt-1">
                    Features will render deterministically as attributes are extracted into the FaceSpec.
                  </p>
                </div>
              </div>

              {/* Mandatory Bottom Watermark Banner */}
              <div className="absolute bottom-0 inset-x-0 bg-red-950/80 backdrop-blur-sm border-t border-red-800/60 py-1 px-3 text-center z-10">
                <p className="text-[10px] font-mono uppercase tracking-wider text-red-200 font-semibold">
                  AI-GENERATED APPROXIMATION - INVESTIGATIVE LEAD ONLY
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: FaceSpec Source-of-Truth Inspector (3 cols) */}
        <section className="lg:col-span-3 flex flex-col bg-slate-900/40">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
            <div className="flex items-center gap-2">
              <Hash className="h-4 w-4 text-cyan-400" />
              <h2 className="font-semibold text-sm">FaceSpec (Source of Truth)</h2>
            </div>
          </div>

          <div className="p-3 bg-slate-950/40 border-b border-slate-800/80 text-xs text-slate-400">
            <strong>Rule 3:</strong> Unstated features stay <code>"unknown"</code>. Images are strictly renderings of this structured state.
          </div>

          <div className="flex-1 p-4 overflow-y-auto space-y-2.5 font-mono text-xs">
            {Object.entries(faceSpec).map(([key, attr]) => (
              <div key={key} className="p-2.5 rounded bg-slate-900/90 border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="text-[11px] text-slate-400 capitalize">{key.replace(/_/g, ' ')}</div>
                  <div className="font-semibold text-slate-200 mt-0.5">
                    {attr.value === "unknown" ? (
                      <span className="text-slate-500 italic">unknown</span>
                    ) : (
                      <span>{attr.value}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                    {Math.round(attr.confidence * 100)}%
                  </span>
                  <button className="text-slate-500 hover:text-slate-300 cursor-pointer p-1">
                    <Lock className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Audit Ledger Inspection Modal */}
      {showAuditModal && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-5 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <ShieldCheck className="h-5 w-5 text-emerald-400" />
                <h3 className="font-semibold text-base">Forensic Audit Ledger (Hash-Chained)</h3>
              </div>
              <button 
                onClick={() => setShowAuditModal(false)}
                className="text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800 text-xs"
              >
                Close
              </button>
            </div>

            <div className="p-4 bg-slate-950/60 border-b border-slate-800 text-xs font-mono">
              <div className="flex items-center gap-2 text-emerald-400">
                <ShieldCheck className="h-4 w-4" />
                <span>{auditResult?.message || "Cryptographic verification active."}</span>
              </div>
              <div className="mt-1 text-slate-400">
                Total Chained Blocks: <span className="text-slate-200">{auditResult?.total_entries ?? 0}</span>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-xs">
              {auditEntries.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  No explicit audit entries recorded in this session yet.
                </div>
              ) : (
                auditEntries.map((e) => (
                  <div key={e.id} className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] space-y-1">
                    <div className="flex items-center justify-between text-slate-400">
                      <span className="font-bold text-cyan-400">#{e.id} {e.action_type}</span>
                      <span>{e.timestamp}</span>
                    </div>
                    <div className="text-slate-300">
                      Endpoint: {e.http_method} {e.endpoint} ({e.status_code})
                    </div>
                    <div className="text-slate-500 truncate">
                      Prev Hash: {e.previous_hash}
                    </div>
                    <div className="text-slate-400 truncate font-semibold">
                      Entry Hash: {e.entry_hash}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
