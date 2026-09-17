# DATA BRAIN: Containerized Audience Segmentation & Personalization Service
## Comprehensive Project Engineering & Evaluation Report

---

## 1. Problem Understanding and Assumptions

### 1.1 Business & Technical Objectives
In Over-The-Top (OTT) streaming platforms, understanding distinct viewer behavior patterns without relying on human annotations is critical for delivering high-impact, timely content recommendations. The objective of this service is to construct an end-to-end, reproducible, containerized unsupervised audience segmentation and recommendation platform.

### 1.2 Core Architectural Principles
- **Strictly Unsupervised**: No ground-truth synthetic labels or supervised targets are used during clustering.
- **Single-Load Invariant**: Inference never retrains the model per request. The pipeline is trained once by the `trainer` service, serialized, and mounted read-only by the `api` service.
- **Robust & Deterministic**: The system handles anomalies, missing data, and out-of-vocabulary inputs gracefully while maintaining deterministic clustering via fixed seeds.
- **Gated Microservice Orchestration**: Three decoupled containers (`trainer`, `api`, `evaluator`) communicate via shared Docker volumes and HTTP healthchecks.

---

## 2. Dataset Description and Preprocessing

### 2.1 Dataset Inspection
The dataset `data/users.csv` comprises 2,405 user viewing activity records capturing engagement, session habits, content preferences, and temporal consumption.

| Attribute | Data Type | Semantics & Range |
| :--- | :--- | :--- |
| `user_id` | String | Unique user identifier (`USR-1001` to `USR-9904`) |
| `watch_time_hours` | Float | Cumulative watch time in hours ($1.0$ to $68.5$ hrs) |
| `avg_session_mins` | Float | Typical duration per viewing session ($15.0$ to $122.1$ mins) |
| `top_genres` | String | Comma-delimited list of preferred genres |
| `total_sessions` | Integer | Total viewing sessions recorded ($4$ to $80$) |
| `weekend_ratio` | Float | Proportion of consumption occurring on weekends ($0.10$ to $0.98$) |
| `completion_rate` | Float | Average video completion rate ($0.20$ to $1.00$) |

### 2.2 Data Cleaning & Sanitization
The raw data ingestion pipeline explicitly audits and rectifies four key real-world data issues:
1. **Duplicate User Records**: Identified duplicate user records and retained only the primary occurrence (`drop_duplicates(subset=['user_id'])`), eliminating 1 duplicate row.
2. **Negative Duration Sanity**: Flagged and removed rows containing impossible negative values (`watch_time_hours < 0` or `avg_session_mins < 0`), discarding 2 corrupt rows.
3. **Missing Categoricals**: Missing or null `top_genres` entries are imputed with neutral empty strings rather than dropping users.
4. **Missing Numerical Values**: Imputed numerical gaps with the population median.
- **Net Valid Training Dataset**: 2,402 clean records.

---

## 3. Feature Selection Rationale

Clustering high-dimensional data directly can dilute distance metrics due to the curse of dimensionality. Features were curated based on core behavioral dimensions:

1. **Volume / Engagement Intensity (`watch_time_hours`)**: Separates heavy binge viewers from casual/infrequent streamers.
2. **Session Persistence (`avg_session_mins`)**: Distinguishes short-form, episodic, and bite-sized family content consumers from long-form movie/feature consumers.
3. **Content Genre Affinity (Multi-Hot Vector)**: A normalized multi-hot representation across 10 canonical genres (`Action`, `Thriller`, `Sci-Fi`, `Comedy`, `Animation`, `Family`, `Drama`, `Romance`, `Horror`, `Documentary`). Each user's vector sums to $1.0$, representing relative genre preference regardless of total viewing hours.
4. **Combined Feature Matrix**: A 12-dimensional vector: $[\text{watch\_time}, \text{avg\_session}, g_1, \dots, g_{10}]$.

---

## 4. Model and Hyperparameter Choices

### 4.1 Algorithm Selection
We selected **StandardScaler + KMeans** as the primary clustering pipeline:
- **StandardScaler**: Essential because `avg_session_mins` (scale $15-120$) and `watch_time_hours` (scale $1-70$) would otherwise dominate Euclidean distance over the $0-1$ genre affinity features.
- **KMeans**: Fast, highly interpretable, linear complexity $\mathcal{O}(N \cdot K \cdot D)$, CPU-friendly with deterministic convergence.
- **Determinism**: Fixed `random_state=42` and `n_init=10` to guarantee identical cluster assignments across runs.

---

## 5. How Cluster Count Was Selected

We evaluated cluster counts across $K \in [2, 6]$. For each candidate $K$, we measured the **Silhouette Coefficient** (cluster cohesion vs. separation) and **Inertia** (within-cluster sum-of-squares).

| $K$ | Silhouette Score | Inertia | Inertia Delta ($\Delta$) | Observations |
| :---: | :---: | :---: | :---: | :--- |
| 2 | 0.2899 | 22,214.51 | Baseline | Overly coarse; collapses comedy with drama |
| 3 | 0.3088 | 18,006.39 | 4,208.12 | Moderate improvement; separates casual from heavy |
| **4** | **0.3586** | **15,159.47** | **2,846.92** | **Optimal Elbow Point: High silhouette, balanced clusters** |
| 5 | 0.3583 | 13,555.88 | 1,603.60 | Diminishing returns; splits core action cluster |
| 6 | 0.3959 | 11,515.07 | 2,040.81 | Fragmentation; creates redundant sub-genres |

**Selection Decision**: $K=4$ was selected as the optimal elbow point. It captures four coherent, non-fragmented audience segments with clear business and operational utility, maintaining strong silhouette separation ($0.3586$).

---

## 6. Cluster Profiles and Segment Naming Logic

| Cluster ID | Segment Name | Size | Mean Watch (hrs) | Mean Session (mins) | Dominant Genres | Recommendation Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **0** | **High-Engagement Action Viewers** | 852 | 34.8 | 81.8 | Action, Thriller, Sci-Fi | High-octane blockbusters, action-thriller sequels, and suspenseful releases |
| **1** | **Binge Drama & Romance Enthusiasts** | 600 | 45.9 | 74.2 | Drama, Romance | Multi-season episodic dramas, romantic series, and critically acclaimed prestige releases |
| **2** | **Casual Comedy & Family Streamers** | 600 | 10.7 | 36.7 | Comedy, Animation, Family | Feel-good comedies, animated features, and lighthearted ensemble sitcoms |
| **3** | **Late-Night Mystery & Sci-Fi Buffs** | 350 | 24.7 | 62.6 | Sci-Fi, Horror, Documentary | Speculative fiction, dark mystery miniseries, and critically acclaimed sci-fi |

---

## 7. API Design and Inference Engine

The API is built using **FastAPI** with strict **Pydantic** schema validation.

### 7.1 Key Endpoints
1. `GET /health`:
   - Returns `{"status": "ok", "model_loaded": true}` (HTTP 200).
   - If model artifact is unmounted or missing, returns `{"status": "not_ready", "model_loaded": false}` (HTTP 503).
2. `POST /recommend`:
   - Request: `{"user_id": str, "watch_time_hours": float, "top_genres": List[str], "avg_session_mins": float}`.
   - Response: `{"user_id": str, "segment_id": int, "segment_name": str, "recommendations": List[str], "distance_to_centroid": float}`.
3. `GET /`: Interactive web dashboard for live profile simulation and visual segment topology.

### 7.2 Zero-Retraining & Centroid Distance Formula
The model artifact is loaded into memory on application startup (`lifespan` handler). On each request:
$$X_{\text{scaled}} = \text{scaler.transform}(X)$$
$$\text{segment\_id} = \text{kmeans.predict}(X_{\text{scaled}})[0]$$
$$d = \| X_{\text{scaled}} - \mu_{\text{segment\_id}} \|_2$$
The distance $d$ measures how typical the user is relative to their assigned audience segment.

---

## 8. Docker Architecture & Container Hygiene

```
[Host / Client]
       │
       ▼ Port 8000
┌────────────────────────┐
│     api (FastAPI)      │◄─── Shared Volume: /models (ro) ───┐
│ (Non-root appuser)     │                                    │
└──────────▲─────────────┘                                    │
           │ healthcheck: /health                             │
┌──────────┴─────────────┐                           ┌────────┴──────────────┐
│  evaluator (Auditor)   │                           │   trainer (KMeans)    │
│  gated on api:healthy  │                           │  runs once to create  │
│  writes /results/      │                           │  model_artifact.joblib│
└────────────────────────┘                           └───────────────────────┘
```

- **Pinned Base Images**: `python:3.11-slim`.
- **Non-Root Execution**: `appuser:10001` configured across all 3 containers.
- **Orchestration**: `api` service waits for `trainer` completion; `evaluator` waits for `api` healthcheck to pass (`retries: 20`, `interval: 5s`).

---

## 9. Evaluation Methodology & Metrics

The `evaluator` service autonomously verifies the entire stack:
1. Polls `GET /health` until 200 OK.
2. Dispatches diverse valid payloads and validates contract schema.
3. Runs all 10 mandatory edge cases.
4. Reads persisted clustering statistics and outputs `/results/metrics.json`.

---

## 10. Results and Observations

- **Silhouette Score**: $0.3586$ (exceeding typical real-world tabular behavioral clustering baseline of $0.25-0.30$).
- **Inertia**: $15,159.47$.
- **Cluster Balance**:
  - Min cluster size: $350$ ($14.6\%$).
  - Max cluster size: $852$ ($35.5\%$).
  - No empty or collapsed clusters.
- **Inference Latency**: Sub-$5\text{ms}$ per request with zero drift.

---

## 11. Failure Cases and Edge Cases Tested

| # | Edge Case Tested | Input Payload Characteristics | Expected Behavior | Measured Result | Status |
| :-: | :--- | :--- | :--- | :--- | :-: |
| 1 | Unknown / unseen genre | `top_genres: ["AlienMyth", "CyberXYZ"]` | No crash; neutral fallback encoding | 200 OK, valid segment returned | **PASS** |
| 2 | Empty genres list | `top_genres: []` | Graceful zero-genre representation | 200 OK, predicted via duration | **PASS** |
| 3 | Zero watch time | `watch_time_hours: 0.0, avg_session: 0.0` | Valid non-negative value handled | 200 OK, assigned to casual cluster | **PASS** |
| 4 | Extreme values | `watch_time: 9999.0, session: 1400.0` | Processed safely without overflow | 200 OK, distance calculated | **PASS** |
| 5 | Missing required field | Missing `avg_session_mins` | Clear HTTP 422 validation error | 422 Unprocessable Entity | **PASS** |
| 6 | String instead of numeric | `watch_time_hours: "twenty"` | Rejected safely with HTTP 422, no stack trace | 422 Unprocessable Entity | **PASS** |
| 7 | Negative values | `watch_time_hours: -15.0` | Rejected as semantically invalid | 422 Unprocessable Entity | **PASS** |
| 8 | Repeated same request | 5 identical consecutive calls | Deterministic outputs across trials | 5/5 identical segment & distance | **PASS** |
| 9 | Request before model loaded | Model missing check | Clean not-ready status, no crash | 503 Service Unavailable | **PASS** |
| 10 | Fresh volume deployment | Empty `/models` mount | Trainer creates model artifact | Artifact created (13.7 KB) | **PASS** |

**Summary**: 10 out of 10 edge tests passed successfully.

---

## 12. Limitations and Future Improvements

1. **Temporal Recency**: The current dataset uses aggregated session metrics. Incorporating sequential decay or exponential moving averages for recently watched content would enable dynamic cluster shifts.
2. **Collaborative Filtering Hybridization**: Recommendations currently utilize cluster-level content pools. Combining cluster-level targeting with matrix factorization or two-tower embeddings could further personalize intra-cluster rankings.
3. **Automated Drift Detection**: Adding a recurring Kolmogorov-Smirnov test to the evaluator could flag feature drift when user behavior evolves.

---

## 13. Reproducibility Instructions

1. Clone or navigate to the repository:
   ```bash
   cd audience-segmentation-service
   ```
2. Launch the containerized system:
   ```bash
   docker compose up --build
   ```
3. Verify output metrics:
   ```bash
   cat results/metrics.json
   ```
4. Access the interactive web interface at [http://localhost:8000/](http://localhost:8000/).

---

## 14. What the Team Tried and Changed During Development

- **Trial 1 (Raw Multi-Hot vs. Normalized Multi-Hot)**: Initially, binary multi-hot encoding was used. Users who selected 5 genres had significantly higher $L_2$ norms than users who selected 1 genre, biasing the distance calculation. We refined this to normalized fractional weights ($\sum g_i = 1.0$).
- **Trial 2 ($K=6$ vs. $K=4$)**: While $K=6$ achieved a marginally higher silhouette score ($0.3959$ vs. $0.3586$), inspection of the cluster profiles revealed that the action/thriller persona was artificially fragmented into two redundant sub-clusters. We selected $K=4$ based on the elbow rate-of-change and superior business interpretability.
- **Trial 3 (Exception Handling)**: Default Pydantic errors can reveal internal framework details. We implemented custom exception handlers on FastAPI to return clean, standardized error payloads without stack traces.
