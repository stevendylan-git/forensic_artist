import { HealthResponse, AuditEntry, AuditVerificationResult } from '../types';

const API_BASE = '/api/v1';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status: ${res.status}`);
  }
  return res.json();
}

export async function fetchAuditEntries(limit = 20): Promise<AuditEntry[]> {
  const res = await fetch(`${API_BASE}/audit/entries?limit=${limit}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch audit entries: ${res.status}`);
  }
  return res.json();
}

export async function verifyAuditLedger(): Promise<AuditVerificationResult> {
  const res = await fetch(`${API_BASE}/audit/verify`);
  if (!res.ok) {
    throw new Error(`Failed to verify audit ledger: ${res.status}`);
  }
  return res.json();
}
