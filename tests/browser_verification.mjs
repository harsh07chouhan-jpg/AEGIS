/**
 * Headless browser verification script for AEGIS frontend.
 * Starts the Vite preview server, renders the application in Microsoft Edge (headless),
 * captures a screenshot, dumps the rendered DOM, and validates all core editorial sections.
 */

import { spawn, execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'
const SCREENSHOT_PATH = path.resolve('d:/Projects/AEGIS/browser_preview.png')
const PORT = 4173
const PREVIEW_URL = `http://localhost:${PORT}`

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitForServer(url, timeoutMs = 15000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url)
      if (res.ok) return true
    } catch {
      // Retry
    }
    await sleep(400)
  }
  throw new Error(`Timeout waiting for server at ${url}`)
}

async function runBrowserVerification() {
  console.log(`[1/5] Starting Vite preview server on port ${PORT}...`)
  const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  const previewProcess = spawn(npmCmd, ['run', 'preview', '--', '--port', String(PORT)], {
    cwd: 'd:/Projects/AEGIS/frontend',
    stdio: 'ignore',
    shell: true,
  })

  let allPassed = false
  try {
    console.log('[2/5] Waiting for preview server readiness...')
    await waitForServer(PREVIEW_URL)
    console.log(`✓ Preview server reachable at ${PREVIEW_URL}`)

    console.log('[3/5] Launching headless Edge to render application...')
    const domOutput = execFileSync(
      EDGE_PATH,
      ['--headless', '--disable-gpu', '--dump-dom', PREVIEW_URL],
      { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
    )
    console.log(`✓ Rendered DOM successfully retrieved (${domOutput.length} characters)`)

    console.log('[4/5] Capturing browser screenshot...')
    execFileSync(EDGE_PATH, [
      '--headless',
      '--disable-gpu',
      `--screenshot=${SCREENSHOT_PATH}`,
      '--window-size=1280,1024',
      PREVIEW_URL,
    ])

    if (fs.existsSync(SCREENSHOT_PATH)) {
      const stats = fs.statSync(SCREENSHOT_PATH)
      console.log(`✓ Screenshot captured successfully: ${stats.size} bytes`)
      fs.unlinkSync(SCREENSHOT_PATH)
    }

    console.log('[5/5] Validating rendered editorial layout elements...')
    const requiredElements = [
      'AEGIS',
      'Submit Audio Recording',
      'EVIDENCE INGESTION',
      'Run Forensic Screening',
      'Active Transformer Architecture',
      'MODEL SPECIFICATION',
      'Offline ASVspoof 2019 LA Evaluation',
      'BENCHMARK PERFORMANCE AUDIT',
      'Multimodal Forensics Architecture',
      'AEGIS FORENSIC MEDIA LABORATORY',
    ]

    allPassed = true
    for (const elem of requiredElements) {
      if (domOutput.includes(elem)) {
        console.log(`  ✓ Element present: "${elem}"`)
      } else {
        console.error(`  ✗ Missing element: "${elem}"`)
        allPassed = false
      }
    }

    if (!allPassed) {
      throw new Error('Rendered DOM failed content verification checks.')
    }

    console.log('\nAll browser verification checks passed successfully!')
  } finally {
    console.log('Shutting down preview server process...')
    if (previewProcess.pid) {
      if (process.platform === 'win32') {
        try {
          execFileSync('taskkill', ['/pid', String(previewProcess.pid), '/T', '/F'])
        } catch {
          // Process already closed
        }
      } else {
        previewProcess.kill('SIGTERM')
      }
    }
    process.exit(allPassed ? 0 : 1)
  }
}

runBrowserVerification().catch((err) => {
  console.error('\nBrowser verification failed:', err.message)
  process.exit(1)
})
