import {
  sanitizeFilename,
  triggerFileDownload,
} from './evidenceReport'

const SCIENTIFIC_NOTICE =
  'AEGIS is an academic forensic screening prototype. ' +
  'Model outputs are uncalibrated and are not probabilities of authenticity. ' +
  'A REAL or FAKE classification is not definitive proof. ' +
  'Independent forensic verification is required.'

const validScore = (value) =>
  typeof value === 'number' &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 1

export function isValidVideoResult(result) {
  return (
    result !== null &&
    typeof result === 'object' &&
    !Array.isArray(result) &&
    ['REAL', 'FAKE'].includes(
      String(result.verdict || '').toUpperCase()
    ) &&
    validScore(result.prob_fake) &&
    result.metadata !== null &&
    typeof result.metadata === 'object' &&
    !Array.isArray(result.metadata)
  )
}

export function generateVideoMarkdownReport(result, timestamp = new Date()) {
  if (!isValidVideoResult(result)) {
    throw new Error('Cannot export an invalid video inference result.')
  }

  const metadata = result.metadata
  const predictions = Array.isArray(metadata.frame_predictions)
    ? metadata.frame_predictions
    : []

  const lines = [
    '# AEGIS — VIDEO FORENSIC SCREENING REPORT',
    '',
    `**Generated:** ${timestamp.toISOString()}`,
    '**Analysis type:** Video manipulation screening',
    '**Inference status:** Completed',
    '',
    '## 1. Model Assessment',
    '',
    `- **Classification:** ${String(result.verdict).toUpperCase()}`,
    `- **Fake-class score:** ${(result.prob_fake * 100).toFixed(2)}%`,
    `- **Model-reported confidence:** ${
      validScore(result.confidence)
        ? `${(result.confidence * 100).toFixed(2)}%`
        : 'Not reported'
    }`,
    `- **Model version:** ${result.model_version || 'Not reported'}`,
    '',
    'Scores are uncalibrated model outputs, not verified probabilities.',
    '',
    '## 2. Video Metadata',
    '',
    `- **Duration:** ${metadata.duration_seconds ?? 'Not reported'} seconds`,
    `- **Frame rate:** ${metadata.fps ?? 'Not reported'} FPS`,
    `- **Frames sampled:** ${metadata.frames_sampled ?? 'Not reported'}`,
    `- **Faces detected:** ${metadata.faces_detected ?? 'Not reported'}`,
    '',
    '## 3. Temporal Attention',
    '',
    'Attention values describe relative model emphasis, not independent authenticity scores.',
    '',
  ]

  if (predictions.length > 0) {
    lines.push('| Frame | Timestamp (s) | Attention |')
    lines.push('| --- | --- | --- |')

    predictions.forEach((item, index) => {
      const time = Number.isFinite(item?.timestamp)
        ? item.timestamp.toFixed(2)
        : 'N/A'

      const attention = validScore(item?.attention)
        ? `${(item.attention * 100).toFixed(2)}%`
        : 'N/A'

      lines.push(`| ${index + 1} | ${time} | ${attention} |`)
    })
  } else {
    lines.push('No frame-level attention data was provided.')
  }

  lines.push(
    '',
    '## 4. Scientific Qualifications',
    '',
    SCIENTIFIC_NOTICE,
    '',
    'Benchmark performance does not guarantee real-world accuracy.',
    '',
    'This report is a screening record, not a certification of media authenticity.',
    ''
  )

  return lines.join('\n')
}

export function generateVideoJsonReport(result, timestamp = new Date()) {
  if (!isValidVideoResult(result)) {
    throw new Error('Cannot export an invalid video inference result.')
  }

  return JSON.stringify(
    {
      aegis_export_version: '1.0.0',
      analysis_type: 'video',
      export_timestamp_iso: timestamp.toISOString(),
      export_service: 'AEGIS Client Forensic Exporter',
      scientific_notice: SCIENTIFIC_NOTICE,
      analysis_result: result,
    },
    null,
    2
  )
}

export function exportVideoEvidence(result, format = 'markdown') {
  if (!isValidVideoResult(result)) {
    throw new Error('No valid completed video analysis is available.')
  }

  if (!['markdown', 'json'].includes(format)) {
    throw new Error(`Unsupported export format: ${format}`)
  }

  const timestamp = new Date()
  const date = timestamp.toISOString().slice(0, 10).replace(/-/g, '')
  const filenameBase = sanitizeFilename(
    result.metadata?.filename || 'video_evidence'
  )

  if (format === 'json') {
    const filename = `AEGIS_${filenameBase}_${date}_video_audit.json`

    triggerFileDownload(
      generateVideoJsonReport(result, timestamp),
      filename,
      'application/json'
    )

    return filename
  }

  const filename = `AEGIS_${filenameBase}_${date}_video_report.md`

  triggerFileDownload(
    generateVideoMarkdownReport(result, timestamp),
    filename,
    'text/markdown'
  )

  return filename
}