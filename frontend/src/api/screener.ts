import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { API_QUERY_KEYS } from "@/constants/api";
import { type ApiResponse, apiClient } from "@/lib/api-client";
import type {
  ScreenerCandidateDetailResponse,
  ScreenerCandidateListResponse,
  ScreenerExportResponse,
  ScreenerRunConfig,
  ScreenerRunDetailResponse,
  ScreenerRunListResponse,
  ScreenerRunResponse,
} from "@/types/screener";

export const useRunScreener = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: Partial<ScreenerRunConfig>) =>
      apiClient.post<ApiResponse<ScreenerRunResponse>>("/screener/run", payload),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: API_QUERY_KEYS.SCREENER.runs,
      });
    },
  });
};

export const useGetScreenerRuns = () => {
  return useQuery({
    queryKey: API_QUERY_KEYS.SCREENER.runs,
    queryFn: () =>
      apiClient.get<ApiResponse<ScreenerRunListResponse>>("/screener/runs"),
    select: (data) => data.data.runs,
  });
};

export const useGetScreenerRun = (runId?: string | null) => {
  return useQuery({
    queryKey: API_QUERY_KEYS.SCREENER.runDetail([runId ?? ""]),
    queryFn: () =>
      apiClient.get<ApiResponse<ScreenerRunDetailResponse>>(
        `/screener/runs/${runId}`,
      ),
    select: (data) => data.data,
    enabled: !!runId,
  });
};

export const useGetScreenerCandidates = (runId?: string | null) => {
  return useQuery({
    queryKey: API_QUERY_KEYS.SCREENER.candidates([runId ?? ""]),
    queryFn: () =>
      apiClient.get<ApiResponse<ScreenerCandidateListResponse>>(
        `/screener/runs/${runId}/candidates`,
      ),
    select: (data) => data.data.candidates,
    enabled: !!runId,
  });
};

export const useGetScreenerCandidateDetail = (
  runId?: string | null,
  ticker?: string | null,
) => {
  return useQuery({
    queryKey: API_QUERY_KEYS.SCREENER.candidateDetail([
      runId ?? "",
      ticker ?? "",
    ]),
    queryFn: () =>
      apiClient.get<ApiResponse<ScreenerCandidateDetailResponse>>(
        `/screener/runs/${runId}/candidates/${ticker}`,
      ),
    select: (data) => data.data.detail,
    enabled: !!runId && !!ticker,
  });
};

export const useExportScreenerCandidates = (runId?: string | null) => {
  return useQuery({
    queryKey: API_QUERY_KEYS.SCREENER.export([runId ?? ""]),
    queryFn: () =>
      apiClient.get<ApiResponse<ScreenerExportResponse>>(
        `/screener/runs/${runId}/export`,
      ),
    select: (data) => data.data,
    enabled: !!runId,
  });
};
