import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  useDeleteScreenerRun,
  useExportScreenerCandidates,
  useGetScreenerCandidateDetail,
  useGetScreenerCandidates,
  useGetScreenerRun,
  useGetScreenerRuns,
  useRunScreener,
} from "@/api/screener";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Trash2 } from "lucide-react";
import type { ScreenerRunConfig } from "@/types/screener";

const FREQUENCY_OPTIONS: Array<ScreenerRunConfig["frequency"]> = [
  "weekly",
  "monthly",
  "quarterly",
];

const formatTimestamp = (value?: string | null) => {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
};

const formatDocRefs = (docRef: Record<string, string | number | number[]>) => {
  const entries = Object.entries(docRef);
  if (entries.length === 0) return null;
  return entries.map(([key, value]) => {
    if (Array.isArray(value)) {
      return `${key}: ${value.join(", ")}`;
    }
    return `${key}: ${value}`;
  });
};

export default function ScreenerPage() {
  const { t } = useTranslation();
  const [frequency, setFrequency] =
    useState<ScreenerRunConfig["frequency"]>("monthly");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const { data: runs = [] } = useGetScreenerRuns();
  const { data: runDetail } = useGetScreenerRun(selectedRunId);
  const { data: candidates = [], isLoading: candidatesLoading } =
    useGetScreenerCandidates(selectedRunId);
  const { data: candidateDetail } = useGetScreenerCandidateDetail(
    selectedRunId,
    selectedTicker,
  );
  const { data: exportData } = useExportScreenerCandidates(selectedRunId);
  const { mutateAsync: runScreener, isPending } = useRunScreener();
  const { mutateAsync: deleteRun, isPending: deletePending } =
    useDeleteScreenerRun();

  useEffect(() => {
    if (runs.length === 0) {
      setSelectedRunId(null);
      setSelectedTicker(null);
      return;
    }
    const hasSelectedRun = runs.some((run) => run.run_id === selectedRunId);
    if (!selectedRunId || !hasSelectedRun) {
      setSelectedRunId(runs[0].run_id);
      setSelectedTicker(null);
    }
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (!selectedTicker && candidates.length > 0) {
      setSelectedTicker(candidates[0].ticker);
    }
  }, [candidates, selectedTicker]);

  useEffect(() => {
    if (!isPending) {
      setProgress(0);
      return;
    }
    setProgress(5);
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) return prev;
        const increment = 0.5 + Math.random() * 1.5;
        return Math.min(prev + increment, 95);
      });
    }, 200);
    return () => clearInterval(interval);
  }, [isPending]);

  const handleRun = async () => {
    const response = await runScreener({ frequency });
    const nextRunId = response.data.run.run_id;
    setSelectedRunId(nextRunId);
    setSelectedTicker(response.data.top_candidates[0]?.ticker ?? null);
  };

  const handleExport = () => {
    if (!exportData) return;
    const blob = new Blob([exportData.content], {
      type: exportData.content_type,
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = exportData.filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleDeleteRun = async (runId: string) => {
    const confirmed = window.confirm(t("screener.history.deleteConfirm"));
    if (!confirmed) return;
    await deleteRun(runId);
    if (runId === selectedRunId) {
      setSelectedRunId(null);
      setSelectedTicker(null);
    }
  };

  const selectedCandidate = useMemo(
    () => candidates.find((item) => item.ticker === selectedTicker),
    [candidates, selectedTicker],
  );

  return (
    <div className="flex size-full flex-col gap-6 overflow-hidden bg-card p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-semibold text-2xl">
            {t("screener.title")}
          </h1>
          <p className="text-muted-foreground text-sm">
            {t("screener.subtitle")}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select
            value={frequency}
            onValueChange={(value) =>
              setFrequency(value as ScreenerRunConfig["frequency"])
            }
          >
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FREQUENCY_OPTIONS.map((option) => (
                <SelectItem key={option} value={option}>
                  {t(`screener.frequency.${option}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button onClick={handleRun} disabled={isPending}>
            {isPending ? t("screener.running") : t("screener.runNow")}
          </Button>
          <Button
            variant="outline"
            onClick={handleExport}
            disabled={!exportData}
          >
            {t("screener.export")}
          </Button>
        </div>
      </div>

      {isPending && (
        <div className="rounded-lg border bg-muted/30 p-3">
          <div className="flex items-center justify-between text-muted-foreground text-xs uppercase tracking-wide">
            <span>{t("common.loading")}</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <Progress value={progress} className="mt-2 h-2 w-full" />
        </div>
      )}

      <div className="grid flex-1 grid-cols-1 gap-6 overflow-hidden lg:grid-cols-[300px_1fr]">
        <Card className="flex h-full flex-col">
          <CardHeader>
            <CardTitle>{t("screener.history.title")}</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto">
            <div className="flex flex-col gap-3">
              {runs.length === 0 && (
                <p className="text-muted-foreground text-sm">
                  {t("screener.history.empty")}
                </p>
              )}
              {runs.map((run) => {
                const isSelected = run.run_id === selectedRunId;
                return (
                  <div
                    key={run.run_id}
                    className={`flex items-start gap-2 rounded-lg border px-3 py-2 text-sm transition ${
                      isSelected
                        ? "border-primary bg-primary/10"
                        : "border-border hover:border-primary/40"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => setSelectedRunId(run.run_id)}
                      className="flex-1 text-left"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium">{run.as_of_date}</span>
                        <Badge variant="secondary">{run.status}</Badge>
                      </div>
                      <div className="mt-1 text-muted-foreground text-xs">
                        {t("screener.history.candidates", {
                          count: run.candidate_count,
                        })}
                      </div>
                    </button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteRun(run.run_id)}
                      disabled={deletePending}
                      title={t("screener.history.delete")}
                      aria-label={t("screener.history.delete")}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <div className="flex h-full flex-col gap-6 overflow-hidden">
          <Card>
            <CardHeader>
              <CardTitle>{t("screener.runSummary.title")}</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm md:grid-cols-2">
              <div>
                <div className="text-muted-foreground">
                  {t("screener.runSummary.universe")}
                </div>
                <div className="font-medium">
                  {runDetail?.run.universe_size ?? "-"}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">
                  {t("screener.runSummary.config")}
                </div>
                <div className="truncate font-medium">
                  {runDetail?.run.config_hash ?? "-"}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">
                  {t("screener.runSummary.git")}
                </div>
                <div className="font-medium">
                  {runDetail?.run.code_git_sha ?? "-"}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground">
                  {t("screener.runSummary.frequency")}
                </div>
                <div className="font-medium">
                  {runDetail?.run.config.frequency ?? "-"}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid flex-1 grid-cols-1 gap-6 overflow-hidden xl:grid-cols-[2fr_1fr]">
            <Card className="flex h-full flex-col overflow-hidden">
              <CardHeader>
                <CardTitle>{t("screener.candidates.title")}</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{t("screener.candidates.rank")}</TableHead>
                      <TableHead>{t("screener.candidates.ticker")}</TableHead>
                      <TableHead>{t("screener.candidates.score")}</TableHead>
                      <TableHead>{t("screener.candidates.wide")}</TableHead>
                      <TableHead>{t("screener.candidates.deep")}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {candidatesLoading && (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center">
                          {t("common.loading")}
                        </TableCell>
                      </TableRow>
                    )}
                    {!candidatesLoading && candidates.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5} className="text-center">
                          {t("screener.candidates.empty")}
                        </TableCell>
                      </TableRow>
                    )}
                    {candidates.map((candidate) => (
                      <TableRow
                        key={candidate.ticker}
                        className={`cursor-pointer ${
                          selectedTicker === candidate.ticker
                            ? "bg-muted/60"
                            : "hover:bg-muted/40"
                        }`}
                        onClick={() => setSelectedTicker(candidate.ticker)}
                      >
                        <TableCell className="font-medium">
                          {candidate.rank}
                        </TableCell>
                        <TableCell>
                          <div className="font-medium">{candidate.ticker}</div>
                          <div className="text-muted-foreground text-xs">
                            {candidate.name}
                          </div>
                        </TableCell>
                        <TableCell>{candidate.total_score.toFixed(2)}</TableCell>
                        <TableCell>{candidate.wide_score.toFixed(2)}</TableCell>
                        <TableCell>{candidate.deep_score.toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            <Card className="flex h-full flex-col overflow-hidden">
              <CardHeader>
                <CardTitle>{t("screener.detail.title")}</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 overflow-auto text-sm">
                {!selectedCandidate && (
                  <p className="text-muted-foreground">
                    {t("screener.detail.empty")}
                  </p>
                )}
                {selectedCandidate && (
                  <div className="flex flex-col gap-4">
                    <div>
                      <div className="text-muted-foreground text-xs uppercase">
                        {t("screener.detail.score")}
                      </div>
                      <div className="mt-1 text-lg">
                        {selectedCandidate.total_score.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs uppercase">
                        {t("screener.detail.rationale")}
                      </div>
                      <p className="mt-1 text-sm">
                        {selectedCandidate.rationale}
                      </p>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs uppercase">
                        {t("screener.detail.evidence")}
                      </div>
                      <ul className="mt-2 space-y-2">
                        {candidateDetail?.evidence.map((item) => (
                          <li key={item.evidence_id} className="rounded-md border p-3">
                            <div className="text-xs text-muted-foreground">
                              {item.type} · {item.source_name}
                            </div>
                            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs">
                              <span>
                                {t("screener.detail.publishedAt")}:{" "}
                                {formatTimestamp(item.published_at)}
                              </span>
                              <span>
                                {t("screener.detail.retrievedAt")}:{" "}
                                {formatTimestamp(item.retrieved_at)}
                              </span>
                            </div>
                            <p className="mt-2 text-sm">{item.quote}</p>
                            <div className="mt-2 flex flex-col gap-1 text-xs text-muted-foreground">
                              {item.source_url && (
                                <a
                                  href={item.source_url}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="text-primary underline-offset-4 hover:underline"
                                >
                                  {t("screener.detail.sourceLink")}
                                </a>
                              )}
                              {formatDocRefs(item.doc_ref)?.map((entry) => (
                                <span key={entry}>{entry}</span>
                              ))}
                            </div>
                          </li>
                        ))}
                        {candidateDetail?.evidence.length === 0 && (
                          <li className="space-y-2 text-sm text-muted-foreground">
                            <p>{t("screener.detail.noEvidence")}</p>
                            <p>{t("screener.detail.noEvidenceHint")}</p>
                          </li>
                        )}
                      </ul>
                    </div>
                    <div>
                      <div className="text-muted-foreground text-xs uppercase">
                        {t("screener.detail.logic")}
                      </div>
                      {!candidateDetail && (
                        <p className="mt-2 text-sm text-muted-foreground">-</p>
                      )}
                      {candidateDetail && candidateDetail.logic_graph.nodes.length === 0 && (
                        <div className="mt-2 space-y-2 text-sm text-muted-foreground">
                          <p>{t("screener.detail.noLogic")}</p>
                          <p>{t("screener.detail.noLogicHint")}</p>
                        </div>
                      )}
                      {candidateDetail && candidateDetail.logic_graph.nodes.length > 0 && (
                        <pre className="mt-2 whitespace-pre-wrap rounded-md bg-muted p-2 text-xs">
                          {JSON.stringify(candidateDetail.logic_graph, null, 2)}
                        </pre>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
