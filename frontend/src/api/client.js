// Typed API client for the Price Truth backend.
//
// A single axios instance targets `API_BASE_URL` (empty in dev so the Vite
// proxy forwards `/api` and `/health` to the backend; a full origin in
// production). All traffic is JSON (Req 14.4) and, in production, HTTPS.
//
// The response-error interceptor unwraps the backend's structured error payload
// ({ error: { code, message, status, details } }, Req 15.3) into a plain
// Error carrying `code`, `status`, `message`, and (when present) `details`, so
// callers can branch on a stable shape regardless of transport-level failures.
import axios from 'axios'
import { API_BASE_URL } from '../config.js'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
})

/**
 * Normalize any axios failure into an Error with a predictable surface.
 * Structured backend errors take precedence; transport errors (timeouts,
 * network) fall back to axios's own code/message.
 */
export function normalizeApiError(error) {
  const structured = error?.response?.data?.error
  const message =
    structured?.message ||
    error?.message ||
    'The request failed. Please try again.'
  const normalized = new Error(message)
  normalized.name = 'ApiError'
  normalized.isApiError = true
  normalized.code = structured?.code ?? error?.code ?? 'UNKNOWN_ERROR'
  normalized.status = structured?.status ?? error?.response?.status ?? 0
  if (structured?.details !== undefined) {
    normalized.details = structured.details
  }
  return normalized
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(normalizeApiError(error)),
)

// --- Endpoint functions -----------------------------------------------------
// Each returns the parsed response body (`response.data`).

/** GET /api/v1/search?q= — product search (Req 1.1, 1.2). */
export async function searchProducts(q) {
  const { data } = await apiClient.get('/api/v1/search', { params: { q } })
  return data
}

/** POST /api/v1/manual-entry — manual product entry (Req 1.6). */
export async function manualEntry(payload) {
  const { data } = await apiClient.post('/api/v1/manual-entry', payload)
  return data
}

/** POST /api/v1/discount-check — genuineness score + SHAP breakdown (Req 2, 3). */
export async function checkDiscount(payload) {
  const { data } = await apiClient.post('/api/v1/discount-check', payload)
  return data
}

/** GET /api/v1/shrinkflation/{id} — pack-size timeline (Req 4). */
export async function getShrinkflation(productId) {
  const { data } = await apiClient.get(
    `/api/v1/shrinkflation/${encodeURIComponent(productId)}`,
  )
  return data
}

/** POST /api/v1/unit-price/compare — per-unit comparison across variants (Req 5). */
export async function compareUnitPrice(variants) {
  const { data } = await apiClient.post('/api/v1/unit-price/compare', {
    variants,
  })
  return data
}

/** GET /api/v1/buy-timing/{category} — category buy-now/wait signal (Req 6). */
export async function getBuyTiming(category) {
  const { data } = await apiClient.get(
    `/api/v1/buy-timing/${encodeURIComponent(category)}`,
  )
  return data
}

/** GET /api/v1/cross-platform/{id} — per-platform prices + best deal (Req 7). */
export async function getCrossPlatform(productId) {
  const { data } = await apiClient.get(
    `/api/v1/cross-platform/${encodeURIComponent(productId)}`,
  )
  return data
}

/** GET /api/v1/dashboard/{id} — composite of all five modules (Req 8). */
export async function getDashboard(productId) {
  const { data } = await apiClient.get(
    `/api/v1/dashboard/${encodeURIComponent(productId)}`,
  )
  return data
}

/** GET /api/v1/data-sources — data sources + limitations disclosure (Req 10.2). */
export async function getDataSources() {
  const { data } = await apiClient.get('/api/v1/data-sources')
  return data
}

/** GET /health — backend + DB/Redis liveness (Req 16.1). */
export async function getHealth() {
  const { data } = await apiClient.get('/health')
  return data
}
