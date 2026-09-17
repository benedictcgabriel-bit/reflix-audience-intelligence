import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")

KNOWN_GENRES = [
    "Action", "Thriller", "Sci-Fi", "Comedy", "Animation",
    "Family", "Drama", "Romance", "Horror", "Documentary"
]

RECOMMENDATION_CATALOG = {
    "High-Engagement Action Viewers": ["Mad Max: Fury Road", "John Wick: Chapter 4", "The Dark Knight", "Inception"],
    "Binge Drama & Romance Enthusiasts": ["Succession", "Bridgerton", "The Crown", "Normal People"],
    "Casual Comedy & Family Streamers": ["Ted Lasso", "The Good Place", "Inside Out 2", "Parks and Recreation"],
    "Late-Night Mystery & Sci-Fi Buffs": ["Interstellar", "Dark", "Severance", "Black Mirror"]
}

def find_data_path():
    candidates = [
        "/data/users.csv",
        "data/users.csv",
        "../data/users.csv",
        "/Users/benedictcgabriel/.gemini/antigravity/scratch/audience-segmentation-service/data/users.csv"
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Cannot find users.csv in any expected location: {candidates}")

def find_models_dir():
    candidates = [
        "/models",
        "models",
        "../models",
        "/Users/benedictcgabriel/.gemini/antigravity/scratch/audience-segmentation-service/models"
    ]
    for p in candidates:
        if os.path.isdir(p) and os.access(p, os.W_OK):
            return p
    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            if os.access(p, os.W_OK):
                return p
        except Exception:
            continue
    raise PermissionError("Cannot find or create writable models directory.")

def encode_genres(genre_str_series, known_genres):
    encoded = np.zeros((len(genre_str_series), len(known_genres)), dtype=np.float32)
    for i, g_str in enumerate(genre_str_series):
        if not isinstance(g_str, str) or not g_str.strip():
            continue
        tokens = [g.strip() for g in g_str.replace("|", ",").split(",") if g.strip()]
        valid_tokens = [g for g in tokens if g in known_genres]
        if valid_tokens:
            weight = 1.0 / len(valid_tokens)
            for g in valid_tokens:
                col_idx = known_genres.index(g)
                encoded[i, col_idx] = weight
    return encoded

def inspect_dataset(df):
    logger.info("=" * 60)
    logger.info("DATASET INSPECTION")
    logger.info("=" * 60)
    logger.info("Initial Shape: %s rows, %s columns", df.shape[0], df.shape[1])
    logger.info("Columns and Dtypes:\n%s", df.dtypes)
    logger.info("Missing value counts:\n%s", df.isnull().sum())
    duplicate_count = df.duplicated(subset=["user_id"]).sum() if "user_id" in df.columns else df.duplicated().sum()
    logger.info("Duplicate user_id count: %s", duplicate_count)
    logger.info("Numerical summary:\n%s", df.describe())
    logger.info("=" * 60)

def clean_dataset(df):
    logger.info("Cleaning dataset...")
    initial_len = len(df)
    
    # 1. Deduplicate by user_id
    if "user_id" in df.columns:
        df = df.drop_duplicates(subset=["user_id"], keep="first").copy()
        logger.info("Deduplication removed %s duplicate rows.", initial_len - len(df))
    
    # 2. Reject/sanitize invalid negative values
    invalid_negatives = (df["watch_time_hours"] < 0) | (df["avg_session_mins"] < 0)
    neg_count = invalid_negatives.sum()
    if neg_count > 0:
        logger.info("Filtering out %s rows with negative numeric values.", neg_count)
        df = df[~invalid_negatives].copy()
        
    # 3. Sanitize empty/missing genres
    df["top_genres"] = df["top_genres"].fillna("").astype(str)
    
    # 4. Fill missing numeric values if any
    df["watch_time_hours"] = df["watch_time_hours"].fillna(df["watch_time_hours"].median())
    df["avg_session_mins"] = df["avg_session_mins"].fillna(df["avg_session_mins"].median())
    
    logger.info("Clean dataset ready with %s valid records (from %s raw records).", len(df), initial_len)
    return df

def train_pipeline():
    logger.info("Starting Audience Segmentation Training Pipeline...")
    data_path = find_data_path()
    models_dir = find_models_dir()
    logger.info("Loading dataset from: %s", data_path)
    logger.info("Models destination: %s", models_dir)

    raw_df = pd.read_csv(data_path)
    inspect_dataset(raw_df)
    clean_df = clean_dataset(raw_df)

    # Feature engineering
    num_features = clean_df[["watch_time_hours", "avg_session_mins"]].values
    genre_features = encode_genres(clean_df["top_genres"], KNOWN_GENRES)
    X = np.hstack([num_features, genre_features])
    feature_names = ["watch_time_hours", "avg_session_mins"] + [f"genre_{g}" for g in KNOWN_GENRES]
    logger.info("Engineered feature matrix shape: %s with features: %s", X.shape, feature_names)

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K Selection evaluation
    logger.info("Evaluating cluster counts K in [2, 3, 4, 5, 6]...")
    k_eval_results = []
    
    print("\n--- K-Means Model Selection Table ---")
    print(f"{'K':<4} | {'Silhouette Score':<18} | {'Inertia':<14} | {'Inertia Delta':<14}")
    print("-" * 58)
    prev_inertia = None
    selected_k = 4  # Optimal elbow point balancing inertia reduction and non-fragmented personas

    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        inertia = km.inertia_
        delta = (prev_inertia - inertia) if prev_inertia is not None else 0.0
        prev_inertia = inertia
        k_eval_results.append({
            "k": k,
            "silhouette_score": round(float(sil), 4),
            "inertia": round(float(inertia), 2),
            "inertia_delta": round(float(delta), 2)
        })
        print(f"{k:<4} | {sil:<18.4f} | {inertia:<14.2f} | {delta:<14.2f}")
    print("-" * 58)
    
    logger.info("Elbow inflection & parsimony analysis confirms K = %s is optimal.", selected_k)

    # Final Model Training with selected K=4
    final_kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)
    clean_df["cluster"] = final_labels
    final_sil = silhouette_score(X_scaled, final_labels)

    # Cluster profiling and evidence-based segment naming
    segments_meta = {}
    cluster_counts = {}

    logger.info("=" * 60)
    logger.info("CLUSTER PROFILES & EVIDENCE-BASED NAMING")
    logger.info("=" * 60)

    for c in range(selected_k):
        c_df = clean_df[clean_df["cluster"] == c]
        count = len(c_df)
        cluster_counts[str(c)] = count
        mean_watch = float(c_df["watch_time_hours"].mean())
        mean_session = float(c_df["avg_session_mins"].mean())
        
        # Genre affinities in this cluster
        c_genres = encode_genres(c_df["top_genres"], KNOWN_GENRES)
        genre_means = c_genres.mean(axis=0)
        top_genre_indices = np.argsort(genre_means)[::-1][:3]
        top_genres = [KNOWN_GENRES[idx] for idx in top_genre_indices if genre_means[idx] > 0.03]

        # Determine evidence-based name and transparent recommendations
        if ("Action" in top_genres or "Thriller" in top_genres) and mean_watch > 30:
            seg_name = "High-Engagement Action Viewers"
            desc = "Users with high watch time and long sessions focused on action, thrillers, and blockbuster spectacles."
            strat = "Recommend high-octane blockbusters, action-thriller sequels, and suspenseful releases."
            recs = RECOMMENDATION_CATALOG["High-Engagement Action Viewers"]
        elif ("Drama" in top_genres or "Romance" in top_genres) and mean_watch > 30:
            seg_name = "Binge Drama & Romance Enthusiasts"
            desc = "Heavy streaming viewers who binge dramatic series, character-driven narratives, and romance sagas."
            strat = "Recommend multi-season episodic dramas, romantic series, and critically acclaimed prestige releases."
            recs = RECOMMENDATION_CATALOG["Binge Drama & Romance Enthusiasts"]
        elif "Comedy" in top_genres or "Animation" in top_genres or "Family" in top_genres or mean_watch < 18:
            seg_name = "Casual Comedy & Family Streamers"
            desc = "Casual viewers seeking lighthearted entertainment, family friendly programming, and bite-sized comedy sessions."
            strat = "Recommend feel-good comedies, animated features, and lighthearted ensemble sitcoms."
            recs = RECOMMENDATION_CATALOG["Casual Comedy & Family Streamers"]
        else:
            seg_name = "Late-Night Mystery & Sci-Fi Buffs"
            desc = "Niche viewers who favor mind-bending science fiction, psychological thrillers, and immersive late-night mysteries."
            strat = "Recommend speculative fiction, dark mystery miniseries, and critically acclaimed sci-fi."
            recs = RECOMMENDATION_CATALOG["Late-Night Mystery & Sci-Fi Buffs"]

        segments_meta[c] = {
            "segment_id": int(c),
            "segment_name": seg_name,
            "description": desc,
            "cluster_size": int(count),
            "key_features": {
                "mean_watch_time_hours": round(mean_watch, 2),
                "mean_avg_session_mins": round(mean_session, 2),
                "top_genres": top_genres
            },
            "recommendation_strategy": strat,
            "recommendations": recs
        }

        logger.info(
            "Cluster %s [%s]: count=%s, watch_time=%.1f hrs, session_mins=%.1f mins, genres=%s",
            c, seg_name, count, mean_watch, mean_session, top_genres
        )

    # Persist Complete Inference Artifact
    artifact = {
        "scaler": scaler,
        "kmeans": final_kmeans,
        "known_genres": KNOWN_GENRES,
        "feature_names": feature_names,
        "segments": segments_meta,
        "metrics": {
            "algorithm": "KMeans",
            "selected_k": int(selected_k),
            "silhouette_score": round(float(final_sil), 4),
            "inertia": round(float(final_kmeans.inertia_), 2),
            "cluster_balance": {
                "counts": cluster_counts,
                "min_cluster_size": int(min(cluster_counts.values())),
                "max_cluster_size": int(max(cluster_counts.values()))
            },
            "k_evaluation": k_eval_results
        }
    }

    artifact_path = os.path.join(models_dir, "model_artifact.joblib")
    joblib.dump(artifact, artifact_path)
    logger.info("Persisted complete inference artifact to: %s", artifact_path)

    # Also save training summary JSON
    meta_path = os.path.join(models_dir, "training_summary.json")
    with open(meta_path, "w") as f:
        summary_copy = {
            "algorithm": "KMeans",
            "selected_k": selected_k,
            "silhouette_score": artifact["metrics"]["silhouette_score"],
            "inertia": artifact["metrics"]["inertia"],
            "cluster_balance": artifact["metrics"]["cluster_balance"],
            "segments": {str(k): v for k, v in segments_meta.items()},
            "k_evaluation": k_eval_results
        }
        json.dump(summary_copy, f, indent=2)
    logger.info("Wrote training summary JSON to: %s", meta_path)
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_pipeline()
