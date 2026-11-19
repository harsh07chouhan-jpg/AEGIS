/**
 * Unit tests for AEGIS evidence report exporter.
 * Validates report generation, field completeness, safety guards, and filename sanitization.
 */

import assert from 'node:assert/strict'
import test from 'node:test'
import {
  sanitizeFilename,
  isValidInferenceResult,
  generateMarkdownReport,
  generateJsonReport,
} from '../frontend/src/utils/evidenceReport.js'

// Realistic regression result fixture matching LA_E_2834763.flac
const VALID_RESULT = {
  status: 'ok',
  input: {
    filename: 'LA_E_2834763.flac',
    fileSizeBytes: 32063,
    inputSha256: 'DCA19033BBCC6EBF52E071BAC8E9778CEAE7E4FF0F390EAFD03DCDB15D694D7E',
    durationSec: 1.426937,
    sampleRate: 16000,
  },
  aggregate: {
    numWindows: 1,
    meanSpoofScore: 0.716103,
    maxSpoofScore: 0.716103,
    scoreUsedForDecision: 'mean_spoof_score',
  },
  decision: {
    label: 'spoof',
    displayText: 'Flagged by the model as likely synthetic (score at or above 0.5).',
    scoreSemantics: 'uncalibrated model score, not a probability',
    rule: 'mean_spoof_score >= 0.5',
  },
  windows: [
    {
      index: 0,
      startSec: 0.0,
      endSec: 1.426937,
      audioSamples: 22831,
      paddedFraction: 0.442603,
      spoofScore: 0.716103,
    },
  ],
  warnings: [
    {
      code: 'zero_padded_window',
      message:
        'At least one window was zero-padded (up to 44%), matching the evaluation policy; padded windows count in the mean.',
    },
  ],
  model: {
    name: 'aegis_audio_transformer_reproducible_v1',
    checkpoint_sha256: 'B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4',
    checkpoint_epoch: 5,
  },
  policy: {
    sample_rate: 16000,
    window_sec: 2.56,
    hop_sec: 1.28,
    decision_threshold: 0.5,
  },
  elapsedSec: 0.3541,
}

const REJECTED_RESULT = {
  status: 'rejected',
  error: {
    code: 'unreadable_audio',
    message: 'Audio could not be decoded.',
  },
}

test('sanitizeFilename removes path traversals and invalid characters', () => {
  assert.equal(sanitizeFilename('../../etc/passwd'), 'passwd')
  assert.equal(sanitizeFilename('C:\\Windows\\System32\\cmd.exe'), 'cmd.exe')
  assert.equal(sanitizeFilename('audio<test>:file?.wav'), 'audio_test__file_.wav')
  assert.equal(sanitizeFilename(''), 'evidence_sample')
  assert.equal(sanitizeFilename(null), 'evidence_sample')
  assert.equal(sanitizeFilename('normal_audio.flac'), 'normal_audio.flac')
})

test('isValidInferenceResult discriminates between success and error states', () => {
  assert.equal(isValidInferenceResult(VALID_RESULT), true)
  assert.equal(isValidInferenceResult(REJECTED_RESULT), false)
  assert.equal(isValidInferenceResult(null), false)
  assert.equal(isValidInferenceResult({ status: 'ok' }), false)
  assert.equal(
    isValidInferenceResult({
      status: 'ok',
      input: { filename: 'test.wav' },
      aggregate: { meanSpoofScore: 'invalid' },
    }),
    false
  )
})

test('generateMarkdownReport throws on rejected or invalid results', () => {
  assert.throws(
    () => generateMarkdownReport(REJECTED_RESULT),
    /Cannot export report/
  )
  assert.throws(
    () => generateMarkdownReport(null),
    /Cannot export report/
  )
})

test('generateMarkdownReport includes all mandatory forensic fields', () => {
  const md = generateMarkdownReport(VALID_RESULT, new Date('2026-10-25T12:00:00Z'))

  // Filename, SHA-256, duration
  assert.ok(md.includes('LA_E_2834763.flac'), 'Filename missing')
  assert.ok(
    md.includes('DCA19033BBCC6EBF52E071BAC8E9778CEAE7E4FF0F390EAFD03DCDB15D694D7E'),
    'Input SHA-256 missing'
  )
  assert.ok(md.includes('1.4269 seconds'), 'Duration missing')

  // Model & Checkpoint
  assert.ok(md.includes('aegis_audio_transformer_reproducible_v1'), 'Model name missing')
  assert.ok(
    md.includes('B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4'),
    'Checkpoint hash missing'
  )
  assert.ok(md.includes('875,906'), 'Parameter count missing')

  // Classification & Scores
  assert.ok(md.includes('SYNTHETIC / SPOOF FLAGGED'), 'Classification missing')
  assert.ok(md.includes('0.716103'), 'Mean spoof score missing')
  assert.ok(md.includes('0.50'), 'Threshold missing')

  // Windows
  assert.ok(md.includes('| 0 | 0.00s – 1.43s |'), 'Window interval missing')

  // Warnings
  assert.ok(md.includes('zero_padded_window'), 'Warning code missing')

  // Processing Time
  assert.ok(md.includes('0.3541 seconds'), 'Elapsed time missing')

  // Limitations & Uncalibrated notice
  assert.ok(md.includes('Uncalibrated'), 'Uncalibrated notice missing')
  assert.ok(
    md.includes('43.28% false-alarm rate on authentic recordings'),
    'Bona fide false-alarm notice missing'
  )
  assert.ok(
    md.includes('MUST NOT be used as proof of media authenticity'),
    'Authenticity notice missing'
  )
})

test('generateJsonReport outputs valid JSON audit envelope', () => {
  const jsonStr = generateJsonReport(VALID_RESULT, new Date('2026-10-25T12:00:00Z'))
  const parsed = JSON.parse(jsonStr)

  assert.equal(parsed.aegis_export_version, '1.0.0')
  assert.equal(
    parsed.verified_checkpoint_sha256,
    'B937B6BB5C95650D361F75FB9FBE4E8BFFC010070F2853D91E36C64D9D8FD7A4'
  )
  assert.equal(parsed.forensic_summary.decision_label, 'spoof')
  assert.equal(parsed.forensic_summary.mean_spoof_score, 0.716103)
  assert.equal(parsed.forensic_summary.warnings_count, 1)
})
