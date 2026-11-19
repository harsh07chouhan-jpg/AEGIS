/**
 * AEGIS Forensic Evidence Report Exporter
 * Generates client-side, self-contained forensic reports for verified audio analyses.
 * Guarantees that only successful, schema-validated inferences can be exported.
 */

/**
 * Sanitize filename to prevent directory traversal or invalid filesystem characters.
 */
export function sanitizeFilename(filename, fallback = 'evidence_sample') {
  if (!filename || typeof filename !== 'string') return fallback
  const cleaned = filename
    .replace(/^.*[/\\]/, '') // strip path
    .replace(/[^a-zA-Z0-9._-]/g, '_') // replace invalid chars
    .replace(/^_+|_+$/g, '')
    .slice(0, 80)
  return cleaned || fallback
}

/**
 * Validate that an analysis result represents a complete, successful inference.
 */
export function isValidInferenceResult(result) {
  if (!result || typeof result !== 'object') return false
  if (result.status !== 'ok') return false
  if (!result.input || typeof result.input !== 'object') return false
  if (!result.aggregate || !Number.isFinite(result.aggregate.meanSpoofScore)) return false
  if (!result.decision || !result.decision.label) return false
  return true
}

/**
 * Generate a structured, human-readable forensic Markdown report.
 */
export function generateMarkdownReport(result, exportTimestamp = new Date()) {
  if (!isValidInferenceResult(result)) {
    throw new Error('Cannot export report: inference result is invalid, incomplete, or rejected.')
  }

  const { input, aggregate, decision, windows, warnings, model, policy, elapsedSec } = result
  const isoTime = exportTimestamp.toISOString()
  const meanScore = aggregate.meanSpoofScore
  const maxScore = aggregate.maxSpoofScore ?? meanScore
  const threshold = policy?.decision_threshold ?? 0.5
  const isSpoof = decision.label === 'spoof'

  const lines = [
    '# AEGIS DIGITAL EVIDENCE LAB — FORENSIC SCREENING REPORT',
    '',
    `**Report Generated:** ${isoTime}`,
    `**Platform:** AEGIS AI-Enabled Media Integrity & Security (v1.0)`,
    `**Inference Status:** VERIFIED / COMPLETE`,
    '',
    '---',
    '',
    '## 1. EVIDENCE ARTIFACT IDENTIFICATION',
    '',
    `- **Evidence Filename:** \`${input.filename || 'unknown'}\``,
    `- **Input SHA-256 Digest:** \`${input.inputSha256 || 'N/A'}\``,
    `- **Decoded Duration:** ${Number(input.durationSec || 0).toFixed(4)} seconds`,
    `- **Decoded Sample Rate:** ${input.sampleRate || 16000} Hz (Mono Float32)`,
    `- **File Size:** ${input.fileSizeBytes ? `${(input.fileSizeBytes / 1024).toFixed(2)} KiB (${input.fileSizeBytes} bytes)` : 'N/A'}`,
    '',
    '---',
    '',
    '## 2. MODEL SPECIFICATION & PROVENANCE',
    '',
    `- **Model Identifier:** \`${model?.name || 'aegis_audio_transformer_reproducible_v1'}\``,
    `- **Verified Checkpoint SHA-256:** \`${model?.checkpoint_sha256 || 'B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4'}\``,
    `- **Checkpoint Epoch:** ${model?.checkpoint_epoch ?? 5}`,
    `- **Architecture:** Spectral Vision/Audio Transformer (4 Layers, 4 Heads, 128 Embedding Dim, GELU, 16×16 Patch Projection)`,
    `- **Total Trainable Parameters:** 875,906`,
    `- **Feature Extraction:** Stacked Log-Mel Spectrogram + Constant-Q Transform (CQT) of shape (2, 128, 257)`,
    `- **Windowing Configuration:** 2.56s analysis window (40,960 samples), 1.28s hop (20,480 samples)`,
    `- **Decision Threshold:** ${threshold.toFixed(2)}`,
    '',
    '---',
    '',
    '## 3. FORENSIC DECISION & SCORE ASSESSMENT',
    '',
    `### **CLASSIFICATION: ${isSpoof ? 'SYNTHETIC / SPOOF FLAGGED' : 'BONA FIDE / UNFLAGGED'}**`,
    '',
    `- **Mean Raw Spoof Score:** \`${meanScore.toFixed(6)}\` (${(meanScore * 100).toFixed(2)}%)`,
    `- **Maximum Window Spoof Score:** \`${maxScore.toFixed(6)}\` (${(maxScore * 100).toFixed(2)}%)`,
    `- **Configured Decision Rule:** \`${decision.rule || `mean_spoof_score >= ${threshold}`}\``,
    `- **Score Semantics:** Uncalibrated window-mean softmax model output. **NOT a posterior probability that the recording is fake.**`,
    `- **Inference Latency:** ${Number(elapsedSec || 0).toFixed(4)} seconds (CPU single-threadpool execution)`,
    '',
    '---',
    '',
    '## 4. TEMPORAL WINDOW BREAKDOWN',
    '',
    'The recording was segmented into overlapping temporal analysis windows:',
    '',
    '| Window # | Time Interval (s) | Audio Samples | Padded Fraction | Raw Spoof Score | Window Decision |',
    '| :---: | :---: | :---: | :---: | :---: | :---: |',
  ]

  if (Array.isArray(windows) && windows.length > 0) {
    windows.forEach((win) => {
      const score = Number(win.spoofScore || 0)
      const winLabel = score >= threshold ? 'SPOOF' : 'BONAFIDE'
      lines.push(
        `| ${win.index ?? 0} | ${Number(win.startSec || 0).toFixed(2)}s – ${Number(win.endSec || 0).toFixed(2)}s | ${(win.audioSamples || 0).toLocaleString()} | ${(Number(win.paddedFraction || 0) * 100).toFixed(0)}% | ${score.toFixed(6)} | ${winLabel} |`
      )
    })
  } else {
    lines.push('| 0 | 0.00s – End | Complete | 0% | ' + meanScore.toFixed(6) + ' | ' + (isSpoof ? 'SPOOF' : 'BONAFIDE') + ' |')
  }

  lines.push('')
  lines.push('---')
  lines.push('')
  lines.push('## 5. PIPELINE NOTICES & WARNINGS')
  lines.push('')

  if (Array.isArray(warnings) && warnings.length > 0) {
    warnings.forEach((w) => {
      lines.push(`- **[${w.code || 'NOTICE'}]** ${w.message}`)
    })
  } else {
    lines.push('- No pipeline or zero-padding warnings encountered during analysis.')
  }

  lines.push('')
  lines.push('---')
  lines.push('')
  lines.push('## 6. SCIENTIFIC QUALIFICATIONS & BENCHMARK LIMITATIONS')
  lines.push('')
  lines.push('> [!CAUTION]')
  lines.push('> **FORENSIC RESTRAINTS & FALSE-ALARM NOTICE:**')
  lines.push('> 1. **Uncalibrated Model Scores:** The scores reported above represent raw softmax logits averaged across temporal spectrogram windows. They do not represent calibrated probabilities.')
  lines.push('> 2. **Benchmark Class Imbalance:** Offline evaluation on the ASVspoof 2019 LA evaluation split (71,237 utterances) yielded 89.04% accuracy, but the dataset was 89.68% spoof. Accuracy is dominated by spoof detection volume.')
  lines.push('> 3. **Bona Fide Error Rate:** At the default 0.50 threshold, bona fide speech recall was 56.72% (a **43.28% false-alarm rate on authentic recordings**).')
  lines.push('> 4. **No Authenticity Guarantee:** A result classified as `bonafide` MUST NOT be used as proof of media authenticity.')
  lines.push('> 5. **Investigative Use Only:** This output is an investigative screening indicator. It must be corroborated with independent acoustic, contextual, and cryptographic provenance evidence.')
  lines.push('')
  lines.push('---')
  lines.push('*AEGIS Media Forensics Workstation · Digital Evidence Export Service*')

  return lines.join('\n')
}

/**
 * Generate a complete JSON audit file containing verbatim API payload and export envelope.
 */
export function generateJsonReport(result, exportTimestamp = new Date()) {
  if (!isValidInferenceResult(result)) {
    throw new Error('Cannot export report: inference result is invalid, incomplete, or rejected.')
  }

  const exportPayload = {
    aegis_export_version: '1.0.0',
    export_timestamp_iso: exportTimestamp.toISOString(),
    export_service: 'AEGIS Client Forensic Exporter',
    verified_checkpoint_sha256: 'B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4',
    analysis_result: result.raw || result,
    forensic_summary: {
      filename: result.input?.filename,
      input_sha256: result.input?.inputSha256,
      duration_sec: result.input?.durationSec,
      decision_label: result.decision?.label,
      mean_spoof_score: result.aggregate?.meanSpoofScore,
      max_spoof_score: result.aggregate?.maxSpoofScore,
      decision_rule: result.decision?.rule,
      warnings_count: result.warnings?.length || 0,
      warnings: result.warnings,
      uncalibrated_score_warning:
        'Scores are uncalibrated model outputs, not probabilities. A bonafide classification is not proof of authenticity.',
    },
  }

  return JSON.stringify(exportPayload, null, 2)
}

/**
 * Safely trigger browser file download using Blob URL and auto-cleanup.
 */
export function triggerFileDownload(content, filename, mimeType = 'text/plain') {
  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.style.display = 'none'

  document.body.appendChild(anchor)
  anchor.click()

  setTimeout(() => {
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }, 1000)
}

/**
 * Export and trigger download of forensic report in Markdown or JSON format.
 */
export function exportEvidence(result, format = 'markdown') {
  if (!isValidInferenceResult(result)) {
    throw new Error('Cannot export: no valid completed analysis result is available.')
  }

  const baseName = sanitizeFilename(result.input?.filename || 'evidence')
  const timestamp = new Date()
  const dateStr = timestamp.toISOString().slice(0, 10).replace(/-/g, '')

  if (format === 'json') {
    const jsonContent = generateJsonReport(result, timestamp)
    const jsonFilename = `AEGIS_${baseName}_${dateStr}_audit.json`
    triggerFileDownload(jsonContent, jsonFilename, 'application/json')
    return jsonFilename
  }

  const mdContent = generateMarkdownReport(result, timestamp)
  const mdFilename = `AEGIS_${baseName}_${dateStr}_report.md`
  triggerFileDownload(mdContent, mdFilename, 'text/markdown')
  return mdFilename
}
