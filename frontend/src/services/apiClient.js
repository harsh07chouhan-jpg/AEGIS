/**
 * AEGIS API Client
 * Interfaces with the local AEGIS FastAPI backend.
 * Provides resilient request handling, structured error propagation,
 * and response schema normalization.
 */

export const API_BASE = (
  import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, status = 0, code = 'api_error', raw = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.raw = raw
  }
}

/**
 * Check backend health and model loading status.
 */
export async function checkBackendHealth(timeoutMs = 5000) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    })

    clearTimeout(timeoutId)
    if (!res.ok) {
      return {
        online: false,
        status: 'degraded',
        modelLoaded: false,
        startupError: `HTTP ${res.status}`,
      }
    }

    const data = await res.json()
    return {
      online: true,
      status: data.status === 'healthy' && data.model_loaded ? 'online' : 'degraded',
      modelLoaded: Boolean(data.model_loaded),
      startupError: data.startup_error || null,
    }
  } catch (err) {
    clearTimeout(timeoutId)
    return {
      online: false,
      status: 'offline',
      modelLoaded: false,
      startupError: err.name === 'AbortError' ? 'Connection timed out' : 'Backend unreachable',
    }
  }
}

/**
 * Fetch verified model specifications, policy, and benchmark metadata.
 */
export async function fetchModelMetadata(timeoutMs = 6000) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(`${API_BASE}/model-info`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    })
    clearTimeout(timeoutId)

    if (!res.ok) {
      throw new ApiError(`Model metadata request failed (${res.status})`, res.status)
    }

    return await res.json()
  } catch (err) {
    clearTimeout(timeoutId)
    if (err instanceof ApiError) throw err
    throw new ApiError(
      err.name === 'AbortError'
        ? 'Model metadata request timed out.'
        : 'Could not fetch model metadata from server.',
      0,
      'network_error',
      err
    )
  }
}

/**
 * Upload an audio file for live transformer deepfake inference.
 * Returns the verified backend JSON response.
 */
export async function analyzeAudioFile(file) {
  if (!file) {
    throw new ApiError('No audio file provided for analysis.', 400, 'missing_file')
  }

  const formData = new FormData()
  formData.append('file', file)

  let res
  try {
    res = await fetch(`${API_BASE}/analyze/audio`, {
      method: 'POST',
      body: formData,
      headers: {
        Accept: 'application/json',
      },
    })
  } catch (err) {
    throw new ApiError(
      'Cannot reach the AEGIS API. Verify that the backend is running at ' + API_BASE,
      0,
      'network_unreachable',
      err
    )
  }

  let data
  try {
    data = await res.json()
  } catch (err) {
    throw new ApiError(
      `Received invalid response from server (HTTP ${res.status}).`,
      res.status,
      'invalid_json',
      err
    )
  }

  if (!res.ok) {
    const code = data?.detail?.code || `http_${res.status}`
    const message = data?.detail?.message || data?.detail || `Inference request failed (HTTP ${res.status}).`
    throw new ApiError(message, res.status, code, data)
  }

  return normalizeInferenceResponse(data)
}

/**
 * Normalizes and guards response fields to ensure robust UI rendering.
 */
export function normalizeInferenceResponse(raw) {
  if (!raw || typeof raw !== 'object') {
    throw new ApiError('Received empty or malformed result from inference service.', 500, 'malformed_payload')
  }

  const status = raw.status || 'unknown'
  const isOk = status === 'ok'
  const isRejected = status === 'rejected'

  // Input metadata
  const input = {
    filename: raw.input?.filename || 'evidence.audio',
    fileSizeBytes: Number(raw.input?.file_size_bytes) || 0,
    inputSha256: raw.input?.input_sha256 || null,
    durationSec: Number(raw.input?.duration_sec) || 0,
    sampleRate: Number(raw.input?.sample_rate) || 16000,
  }

  // Aggregate results
  const meanSpoofScore = Number(raw.aggregate?.mean_spoof_score)
  const maxSpoofScore = Number(raw.aggregate?.max_spoof_score)
  const numWindows = Number(raw.aggregate?.num_windows) || (raw.windows?.length ?? 0)

  // Decision
  const label = raw.decision?.label || (isOk ? (meanSpoofScore >= 0.5 ? 'spoof' : 'bonafide') : null)
  const displayText = raw.decision?.display_text || ''
  const scoreSemantics = raw.decision?.score_semantics || 'uncalibrated model score, not a probability'
  const decisionRule = raw.decision?.rule || 'mean_spoof_score >= 0.5'

  // Warnings normalized to { code, message }
  const warnings = Array.isArray(raw.warnings)
    ? raw.warnings.map((w) => {
        if (typeof w === 'string') return { code: 'notice', message: w }
        if (w && typeof w === 'object') return { code: w.code || 'warning', message: w.message || JSON.stringify(w) }
        return { code: 'warning', message: String(w) }
      })
    : []

  // Windows array
  const windows = Array.isArray(raw.windows)
    ? raw.windows.map((win, idx) => ({
        index: win.index ?? idx,
        startSec: Number(win.start_sec) || 0,
        endSec: Number(win.end_sec) || 0,
        audioSamples: Number(win.audio_samples) || 0,
        paddedFraction: Number(win.padded_fraction) || 0,
        spoofScore: Number(win.spoof_score) || 0,
      }))
    : []

  // Error block for rejected / error states
  const error = raw.error
    ? {
        code: raw.error.code || 'rejected',
        message: raw.error.message || 'File could not be analysed.',
      }
    : null

  return {
    status,
    isOk,
    isRejected,
    input,
    aggregate: {
      meanSpoofScore: Number.isFinite(meanSpoofScore) ? meanSpoofScore : null,
      maxSpoofScore: Number.isFinite(maxSpoofScore) ? maxSpoofScore : null,
      numWindows,
      scoreUsedForDecision: raw.aggregate?.score_used_for_decision || 'mean_spoof_score',
    },
    decision: {
      label,
      displayText,
      scoreSemantics,
      rule: decisionRule,
    },
    windows,
    warnings,
    error,
    model: raw.model || null,
    policy: raw.policy || null,
    referencePerformance: raw.reference_performance || null,
    limitations: Array.isArray(raw.limitations) ? raw.limitations : [],
    runtime: raw.runtime || null,
    elapsedSec: Number(raw.elapsed_sec) || 0,
    raw,
  }
}
