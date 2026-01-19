export interface ScreenerRunConfig {
  frequency: "weekly" | "monthly" | "quarterly";
  style?: string | null;
  risk?: string | null;
  industry?: string | null;
  size?: string | null;
  liquidity_min?: number | null;
  top_k?: number | null;
  top_n?: number | null;
}

export interface ScreenerScoreBreakdown {
  total_score: number;
  components: Record<string, number>;
  top_evidence_ids: string[];
}

export interface ScreenerRunLogStep {
  name: string;
  status: "completed" | "skipped" | "unverified";
  started_at: string;
  ended_at: string;
  outputs: string[];
  notes?: string | null;
}

export interface ScreenerRunLog {
  run_id: string;
  run_timestamp_utc: string;
  data_cutoff: string;
  steps: ScreenerRunLogStep[];
}

export interface ScreenerCandidate {
  ticker: string;
  name: string;
  wide_score: number;
  deep_score: number;
  risk_score: number;
  total_score: number;
  rank: number;
  score_breakdown: ScreenerScoreBreakdown;
  rationale: string;
}

export interface ScreenerRunMeta {
  run_id: string;
  as_of_date: string;
  run_timestamp_utc?: string | null;
  data_cutoff?: string | null;
  started_at: string;
  ended_at: string;
  config_hash: string;
  code_git_sha: string;
  data_snapshot_hash: string;
  universe_size: number;
  status: string;
  config: ScreenerRunConfig;
  run_log?: ScreenerRunLog | null;
}

export interface ScreenerRunSummary {
  run_id: string;
  as_of_date: string;
  status: string;
  started_at: string;
  ended_at: string;
  candidate_count: number;
  frequency: string;
}

export interface ScreenerEvidence {
  evidence_id: string;
  type: string;
  ticker: string;
  published_at: string;
  retrieved_at: string;
  source_title?: string | null;
  source_name: string;
  source_url: string;
  publisher?: string | null;
  reliability_level?: string | null;
  doc_ref: Record<string, string | number | number[]>;
  quote: string;
  structured: Record<string, string | number>;
  sha256: string;
}

export interface LogicGraphNode {
  id: string;
  type: string;
  name: string;
  value?: number | null;
  unit?: string | null;
  as_of?: string | null;
  evidence_id?: string | null;
}

export interface LogicGraphEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

export interface LogicGraph {
  nodes: LogicGraphNode[];
  edges: LogicGraphEdge[];
}

export interface ScreenerCandidateDetail {
  candidate: ScreenerCandidate;
  evidence: ScreenerEvidence[];
  logic_graph: LogicGraph;
}

export interface ScreenerRunResponse {
  run: ScreenerRunMeta;
  candidate_count: number;
  top_candidates: ScreenerCandidate[];
}

export interface ScreenerRunListResponse {
  runs: ScreenerRunSummary[];
}

export interface ScreenerRunDetailResponse {
  run: ScreenerRunMeta;
  evaluation: Record<string, Record<string, number>>;
}

export interface ScreenerCandidateListResponse {
  run_id: string;
  candidates: ScreenerCandidate[];
}

export interface ScreenerCandidateDetailResponse {
  run_id: string;
  detail: ScreenerCandidateDetail;
}

export interface ScreenerExportResponse {
  run_id: string;
  filename: string;
  content_type: string;
  content: string;
}

export interface ScreenerRunDeleteResponse {
  run_id: string;
}
