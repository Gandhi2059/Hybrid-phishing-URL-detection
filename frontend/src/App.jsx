import { useState, useEffect } from 'react';
import './index.css';

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error(`Server Error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the detection API. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (score) => {
    if (score > 0.6) return '#ef4444'; // Danger
    if (score > 0.3) return '#f59e0b'; // Warning
    return '#10b981'; // Success
  };

  const circumference = 2 * Math.PI * 36; // r=36

  return (
    <div className="container">
      <header>
        <h1>ShieldPhish Engine</h1>
        <p className="subtitle">Real-Time ML, Rules & Threat Intel Detection</p>
      </header>

      <form className="search-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <input
            type="url"
            className="url-input"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/login"
            required
          />
        </div>
        <button 
          type="submit" 
          className={`submit-btn ${loading ? 'loading' : ''}`}
          disabled={loading || !url.trim()}
        >
          <span className="loader"></span>
          {loading ? 'Scanning...' : 'Analyze URL'}
        </button>
      </form>

      {error && <div className="error-msg">{error}</div>}

      {result && (
        <div className="results-card">
          <div className="result-header">
            <div className={`verdict ${result.prediction.toLowerCase() === 'phishing' ? 'phishing' : 'legitimate'}`}>
              <div className="verdict-icon">
                {result.prediction.toLowerCase() === 'phishing' ? '🚨' : '✅'}
              </div>
              <div>
                <div className="stat-label">Verdict</div>
                <div className={`verdict-text ${result.prediction.toLowerCase() === 'phishing' ? 'phishing-text' : 'legit-text'}`}>
                  {result.prediction.toUpperCase()}
                </div>
              </div>
            </div>
            
            <div className="risk-meter">
              <div className="stat-label">Risk Profile</div>
              <div className="risk-circle">
                <svg viewBox="0 0 80 80">
                  <circle cx="40" cy="40" r="36" className="risk-circle-bg" />
                  <circle 
                    cx="40" cy="40" r="36" 
                    className="risk-circle-progress"
                    style={{ 
                      stroke: getRiskColor(result.final_score),
                      strokeDasharray: circumference,
                      strokeDashoffset: circumference - (circumference * result.final_score)
                    }} 
                  />
                </svg>
                <div className="risk-value">
                  {Math.round(result.final_score * 100)}%
                </div>
              </div>
            </div>
          </div>

          <div className="breakdown-grid">
            <div className="stat-box">
              <div className="stat-label">ML Model Confidence</div>
              <div className="stat-value">{Math.round(result.ml_probability * 100)}%</div>
            </div>
            <div className="stat-box">
              <div className="stat-label">Heuristic Penalty</div>
              <div className="stat-value" style={{color: result.rule_score > 0 ? '#f59e0b' : 'inherit'}}>
                +{result.rule_score.toFixed(2)} pts
              </div>
            </div>
          </div>
          
          {/* External Threat Intel */}
          {result.threat_intel && (
            <div className="threat-intel-section" style={{marginBottom: '24px', padding: '16px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--panel-border)'}}>
              <h3 style={{marginBottom: '8px', fontSize: '1.05rem'}}>🌐 External Threat Intel</h3>
              <div style={{display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.95rem'}}>
                <span>DNS Status:</span>
                {result.threat_intel.dns_resolves ? (
                  <span style={{color: 'var(--success-color)'}}>✅ Resolves Active</span>
                ) : (
                  <span style={{color: 'var(--danger-color)'}}>❌ Unreachable (High Risk)</span>
                )}
              </div>
              <div style={{fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '4px'}}>
                {result.threat_intel.domain_intel}
              </div>
            </div>
          )}

          <div className="breakdown-grid" style={{marginBottom: 0}}>
            {/* Rule Engine */}
            <div className="rules-section" style={{margin: 0}}>
              <h3>🔍 Rule Engine Triggers</h3>
              {result.rule_details && result.rule_details.length > 0 ? (
                <ul className="rules-list">
                  {result.rule_details.map((rule, idx) => (
                    <li key={idx}>{rule}</li>
                  ))}
                </ul>
              ) : (
                <p className="no-rules">No suspicious heuristic patterns detected.</p>
              )}
            </div>
            
            {/* XAI Attribution */}
            <div className="rules-section" style={{margin: 0}}>
              <h3>🧠 XAI Explainability (Top Reasons)</h3>
              {result.xai_reasons && result.xai_reasons.length > 0 ? (
                <ul className="rules-list">
                  {result.xai_reasons.map((reason, idx) => (
                    <li key={idx} style={{backgroundColor: 'rgba(59, 130, 246, 0.1)', borderLeftColor: 'var(--primary-color)'}}>{reason}</li>
                  ))}
                </ul>
              ) : (
                <p className="no-rules">ML model did not pinpoint strong malicious features.</p>
              )}
            </div>
          </div>
          
        </div>
      )}
    </div>
  );
}

export default App;
