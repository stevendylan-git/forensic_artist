# Forensic Face Composite Generator (PS-01) - Architecture Documentation

## 1. System Mission & Core Principles

The Forensic Face Composite Generator enables an investigator and an eyewitness to collaborate iteratively on building a facial composite sketch while protecting evidentiary integrity:

1. **FaceSpec as the Single Source of Truth**: The rendered image is merely a visual projection of a structured `FaceSpec` model. The AI never invents or hallucinates features unmentioned by the witness. Every unstated attribute defaults to `"unknown"`.
2. **Cognitive Interview Protocol**: AI dialogue follows strict anti-leading principles. It never asks leading questions (e.g. "Did he have a crooked nose?") and instead relies on open recall ("Tell me about the nose") or objective candidate grids.
3. **Region-Scoped Locality**: Modifying an eye, jaw, or nose must not alter locked or untouched regions. Masks enforce pixel-identical integrity outside the edit boundary.
4. **Tamper-Evident Audit Ledger**: Every access, turn, generation, rejection, and modification is appended to an immutable, cryptographically chained SHA-256 ledger.
5. **Evidentiary Disclaimer & Watermarking**: All outputs carry visible and steganographic/metadata watermarking designating the composite as an investigative approximation.

## 2. Component Topology

```
[ Browser / Witness UI (React + TS + Tailwind) ]
                     │  (HTTP / JSON - No Secrets)
                     ▼
[ FastAPI Backend Gateway ]
   ├── Audit Logging Middleware (Logs all requests/responses/denials)
   ├── Cognitive Interview Agent (Anti-leading prompt engineering)
   ├── Safety Critic (Detects & blocks real named individuals)
   ├── FaceSpec Orchestrator (Pydantic validation)
   ├── Typed Gemini Client (google-genai SDK with retries & timeouts)
   ├── Locality & Compositing Engine (Landmarks & Alpha blending)
   └── Cryptographic Audit Ledger (Append-only SHA-256 hash chaining)
         │                          │
         ▼                          ▼
  [ PostgreSQL DB ]           [ Google GenAI API ]
  (Cases, FaceSpecs,          (Text & Imagen Models)
   Audit Log Chunks)
```

## 3. Audit Hash Chain Specification

Each audit entry computes:
$$\text{entry\_hash} = \text{SHA-256}(\text{id} \parallel \text{timestamp} \parallel \text{action\_type} \parallel \text{payload\_hash} \parallel \text{previous\_hash})$$

- Entry #1 uses genesis previous hash: `0000000000000000000000000000000000000000000000000000000000000000`.
- Tampering with any log entry invalidates all subsequent entries in the chain.
