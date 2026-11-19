export function BenchmarkPanel({ referencePerformance }) {
  // Authoritative figures from evaluation_summary.json (71,237 utterances)
  const accuracy = ((referencePerformance?.accuracy ?? 0.8903519) * 100).toFixed(2)
  const balancedAcc = ((referencePerformance?.balanced_accuracy ?? 0.7473935) * 100).toFixed(2)
  const rocAuc = (referencePerformance?.roc_auc ?? 0.8845935).toFixed(3)
  const eer = ((referencePerformance?.eer_approx ?? 0.1908947) * 100).toFixed(2)
  const spoofPrecision = ((referencePerformance?.spoof_prediction_precision_at_0_5 ?? 0.9490206) * 100).toFixed(1)
  const spoofRecall = ((referencePerformance?.spoof_recall_at_0_5 ?? 0.9275539) * 100).toFixed(1)
  const bonafideRecall = ((referencePerformance?.bonafide_recall_at_0_5 ?? 0.5672332) * 100).toFixed(1)

  return (
    <section className="editorial-section" id="section-benchmark" aria-labelledby="benchmark-heading">
      <div className="section-header">
        <span className="section-number">04</span>
        <div className="section-header-text">
          <span className="section-kicker">BENCHMARK PERFORMANCE AUDIT</span>
          <h2 id="benchmark-heading">Offline ASVspoof 2019 LA Evaluation</h2>
          <p className="section-description">
            Empirical evaluation measured across the complete ASVspoof 2019 Logical Access evaluation partition (71,237 total utterances).
          </p>
        </div>
      </div>

      <div className="benchmark-stat-row">
        <div className="stat-card">
          <span className="stat-label">Overall Accuracy</span>
          <strong className="stat-numeral">{accuracy}%</strong>
          <span className="stat-caption">Dominated by spoof class volume</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Balanced Accuracy</span>
          <strong className="stat-numeral">{balancedAcc}%</strong>
          <span className="stat-caption">Unweighted mean of both classes</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">ROC AUC</span>
          <strong className="stat-numeral">{rocAuc}</strong>
          <span className="stat-caption">Area under ROC curve</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Approximate EER</span>
          <strong className="stat-numeral">{eer}%</strong>
          <span className="stat-caption">Equal error rate estimate</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Spoof Precision</span>
          <strong className="stat-numeral">{spoofPrecision}%</strong>
          <span className="stat-caption">Recall: {spoofRecall}%</span>
        </div>

        <div className="stat-card stat-highlight">
          <span className="stat-label">Bona Fide Recall @ 0.50</span>
          <strong className="stat-numeral numeral-terracotta">{bonafideRecall}%</strong>
          <span className="stat-caption">43.28% False-Alarm Rate on Authentic</span>
        </div>
      </div>

      {/* Confusion Matrix Table */}
      <div className="editorial-matrix-block">
        <div className="matrix-heading">
          <h4>Evaluation Confusion Matrix (71,237 Utterances)</h4>
          <span>Rows: Ground Truth · Columns: Model Classification</span>
        </div>

        <table className="editorial-data-table matrix-table">
          <thead>
            <tr>
              <th>Ground Truth</th>
              <th>Predicted Bona Fide (&lt; 0.50)</th>
              <th>Predicted Spoof (&ge; 0.50)</th>
              <th>Total Count</th>
              <th>Class Recall</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Actual Bona Fide</strong></td>
              <td>4,172 (True Negative)</td>
              <td className="cell-flagged"><strong>3,183 (False Positive)</strong></td>
              <td>7,355</td>
              <td><strong>56.72%</strong></td>
            </tr>
            <tr>
              <td><strong>Actual Spoof</strong></td>
              <td>4,628 (False Negative)</td>
              <td>59,254 (True Positive)</td>
              <td>63,882</td>
              <td><strong>92.76%</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="editorial-callout callout-warning">
        <div className="callout-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <strong>Class-Imbalance Audit & Operational Caveat</strong>
        </div>
        <p>
          The evaluation partition is heavily skewed: <strong>89.68% of samples are synthetic</strong> (63,882 spoof vs. 7,355 bona fide).
          While overall accuracy is high (89.04%), 3,183 out of 7,355 genuine recordings were incorrectly flagged as synthetic at the default 0.50 threshold.
          In real-world populations where authentic speech predominates, the operational false-alarm rate will be substantially higher.
          A negative finding (unflagged) cannot be treated as certification of authenticity.
        </p>
      </div>
    </section>
  )
}
