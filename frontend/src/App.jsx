import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  //state management
  const [teams, setTeams] = useState([]); //list of avaialable teams for backend
  const [homeTeam, setHomeTeam] = useState(""); //selected home team ID/Name
  const [awayTeam, setAwayTeam] = useState(""); //selected away team ID/Name
  const [prediction, setPrediction] = useState(null); //prediction results
  const [metrics, setMetrics] = useState(null); //model performance metrics
  const [featureImportance, setFeatureImportance] = useState([]); //top features driving model predictions
  const [loading, setLoading] = useState(false); //leading state for async operations
  const [error, setError] = useState(""); //error message state

  //itialize data on component mount
  useEffect(() => {
    fetchTeams();
    fetchModelMetrics();
    fetchFeatureImportance();
  }, []);

  //fetches the list of unique teams available in the dataset
  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API_URL}/teams`);
      setTeams(response.data.teams);
    } catch (err) {
      setError("Could not load teams. Make sure the FastAPI backend is running.");
    }
  };

  //fetches global model perfomance metrics
  const fetchModelMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/model-metrics`);
      setMetrics(response.data);
    } catch (err) {
      console.error("Could not load model metrics.");
    }
  };

  //fethces the top 5 features influencing the prediction model
  const fetchFeatureImportance = async () => {
    try {
      const response = await axios.get(`${API_URL}/feature-importance`);
      setFeatureImportance(response.data.features.slice(0, 5));
    } catch (err) {
      console.error("Could not load feature importance.");
    }
  };

  //validates selection and submits match data to the backend for prediction
  const handlePredict = async () => {
    setError("");
    setPrediction(null);

    //ensures both teams are selected and are different teams
    if (!homeTeam || !awayTeam) {
      setError("Please select both teams.");
      return;
    }

    if (homeTeam === awayTeam) {
      setError("Please select two different teams.");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        home_team: homeTeam,
        away_team: awayTeam,
      });

      setPrediction(response.data);
    } catch (err) {
      setError("Prediction failed. Make sure the backend is running and both teams exist.");
    } finally {
      setLoading(false);
    }
  };

  //formats probability values as percentages for display
  const formatPercent = (value) => {
    if (value === undefined || value === null) {
      return "0%";
    }

    return `${value}%`;
  };

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">AI Women&apos;s Soccer Analytics</p>
        <h1>Women&apos;s Soccer  Match Predictor</h1>
        <p>
          A machine learning match predictor for women’s soccer using European league
  and NWSL performance data.
        </p>
      </header>

      <main className="layout">
        {/* Prediction Interface Section */}
        <section className="card prediction-card">
          <h2>Predict a Match</h2>
          <p className="section-subtitle">
            Choose a home team and away team to generate win, draw, and loss probabilities.
          </p>

          <div className="selectors">
            {/* Home Team Selection */}
            <div className="field">
              <label>Home Team</label>
              <select value={homeTeam} onChange={(e) => setHomeTeam(e.target.value)}>
                <option value="">Select home team</option>
                {teams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </div>

            {/* Away Team Selection */}
            <div className="field">
              <label>Away Team</label>
              <select value={awayTeam} onChange={(e) => setAwayTeam(e.target.value)}>
                <option value="">Select away team</option>
                {teams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button onClick={handlePredict} disabled={loading}>
            {loading ? "Predicting..." : "Predict Match"}
          </button>

          {error && <p className="error">{error}</p>}

          {/* Result Display Section */}
          {prediction && (
            <section className="results">
              <div className="match-title">
                <h2>
                  {prediction.home_team} vs {prediction.away_team}
                </h2>
                <span>{prediction.prediction}</span>
              </div>

              <div className="confidence-box">
                <p>Model Confidence</p>
                <h3>{prediction.confidence}%</h3>
              </div>

              <div className="probability-grid">
                <div className="probability-card">
                  <p>Home Win</p>
                  <h3>{formatPercent(prediction.probabilities.HOME_WIN)}</h3>
                </div>

                <div className="probability-card">
                  <p>Draw</p>
                  <h3>{formatPercent(prediction.probabilities.DRAW)}</h3>
                </div>

                <div className="probability-card">
                  <p>Away Win</p>
                  <h3>{formatPercent(prediction.probabilities.AWAY_WIN)}</h3>
                </div>
              </div>

              <div className="expected-goals">
                <div>
                  <p>{prediction.home_team} Expected Goals</p>
                  <h3>{prediction.expected_goals.home_team}</h3>
                </div>

                <div>
                  <p>{prediction.away_team} Expected Goals</p>
                  <h3>{prediction.expected_goals.away_team}</h3>
                </div>
              </div>

              <div className="factors">
                <h3>Key Factors</h3>
                <ul>
                  {prediction.key_factors.map((factor, index) => (
                    <li key={index}>{factor}</li>
                  ))}
                </ul>
              </div>

              <div className="comparison">
                <h3>Team Comparison</h3>
                <div className="comparison-grid">
                  <div>
                    <h4>{prediction.home_team}</h4>
                    <p>Avg Goals For: {prediction.team_comparison.home.avg_goals_for}</p>
                    <p>Avg Goals Against: {prediction.team_comparison.home.avg_goals_against}</p>
                    <p>Avg xG For: {prediction.team_comparison.home.avg_xg_for}</p>
                    <p>Win Rate: {prediction.team_comparison.home.win_rate}%</p>
                  </div>

                  <div>
                    <h4>{prediction.away_team}</h4>
                    <p>Avg Goals For: {prediction.team_comparison.away.avg_goals_for}</p>
                    <p>Avg Goals Against: {prediction.team_comparison.away.avg_goals_against}</p>
                    <p>Avg xG For: {prediction.team_comparison.away.avg_xg_for}</p>
                    <p>Win Rate: {prediction.team_comparison.away.win_rate}%</p>
                  </div>
                </div>
              </div>
            </section>
          )}
        </section>

        <aside className="side-panel">
          <section className="card">
            <h2>Model Performance</h2>

            {metrics ? (
              <div className="metric-list">
                <div>
                  <span>Accuracy</span>
                  <strong>{Math.round(metrics.accuracy * 100)}%</strong>
                </div>
                <div>
                  <span>Precision</span>
                  <strong>{Math.round(metrics.precision * 100)}%</strong>
                </div>
                <div>
                  <span>Recall</span>
                  <strong>{Math.round(metrics.recall * 100)}%</strong>
                </div>
                <div>
                  <span>F1 Score</span>
                  <strong>{Math.round(metrics.f1_score * 100)}%</strong>
                </div>
              </div>
            ) : (
              <p className="muted">Model metrics unavailable.</p>
            )}
          </section>

          <section className="card">
            <h2>Top Model Features</h2>

            {featureImportance.length > 0 ? (
              <div className="feature-list">
                {featureImportance.map((item) => (
                  <div key={item.feature}>
                    <span>{item.feature.replaceAll("_", " ")}</span>
                    <strong>{item.importance.toFixed(3)}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">Feature importance unavailable.</p>
            )}
          </section>
        </aside>
      </main>
    </div>
  );
}

export default App;