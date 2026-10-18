"""
Collaborative Filtering Engine using Matrix Factorization (Truncated SVD).
Learns latent factor embeddings from the Project-Subcontractor historical evaluation matrix.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import joblib

from config import MODELS_DIR, RECOMMENDER_CONFIG

class CollaborativeRecommender:
    def __init__(self, n_components=RECOMMENDER_CONFIG["svd_n_components"]):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=RECOMMENDER_CONFIG["svd_random_state"])
        self.rating_matrix = None
        self.predicted_matrix_df = None
        self.global_mean = 3.5
        self.sub_means = {}
        self.is_fitted = False
        
    def fit(self, evaluations_df, all_subs_df, all_prjs_df):
        """
        Builds sparse interaction matrix of Projects x Subcontractors,
        de-means ratings, applies TruncatedSVD, and reconstructs predicted ratings.
        """
        # Ensure complete grid of known projects and subcontractors
        all_subs = all_subs_df["subcontractor_id"].unique()
        all_prjs = all_prjs_df["project_id"].unique()
        
        # Aggregate any duplicate evaluations by mean
        ratings = evaluations_df.groupby(["project_id", "subcontractor_id"])["rating"].mean().reset_index()
        
        self.global_mean = float(ratings["rating"].mean())
        self.sub_means = ratings.groupby("subcontractor_id")["rating"].mean().to_dict()
        
        # Pivot into interaction matrix: rows=projects, cols=subcontractors
        pivot = ratings.pivot(index="project_id", columns="subcontractor_id", values="rating")
        
        # Reindex to encompass all known entities
        pivot = pivot.reindex(index=all_prjs, columns=all_subs)
        
        # Center ratings by subcontractor mean for SVD (fill missing with subcontractor mean or global mean)
        filled_matrix = pivot.copy()
        for col in filled_matrix.columns:
            mean_val = self.sub_means.get(col, self.global_mean)
            filled_matrix[col] = filled_matrix[col].fillna(mean_val)
            
        # Fit SVD
        # Adjust n_components if matrix dimension is small
        actual_components = min(self.n_components, min(filled_matrix.shape) - 1)
        if actual_components < 2:
            actual_components = 2
        self.svd = TruncatedSVD(n_components=actual_components, random_state=RECOMMENDER_CONFIG["svd_random_state"])
        
        reduced = self.svd.fit_transform(filled_matrix.values)
        reconstructed = self.svd.inverse_transform(reduced)
        
        # Clip reconstructed ratings to valid [1.0, 5.0] range
        reconstructed = np.clip(reconstructed, 1.0, 5.0)
        
        self.predicted_matrix_df = pd.DataFrame(
            reconstructed,
            index=filled_matrix.index,
            columns=filled_matrix.columns
        )
        self.is_fitted = True
        
        # Persist model
        joblib.dump({
            "svd": self.svd,
            "global_mean": self.global_mean,
            "sub_means": self.sub_means,
            "predicted_matrix": self.predicted_matrix_df
        }, MODELS_DIR / "collaborative_svd.pkl")
        
        print(f"[CollaborativeSVD] Fitted on {len(all_prjs)} projects x {len(all_subs)} subcontractors. "
              f"Explained Variance Ratio: {float(np.sum(self.svd.explained_variance_ratio_)):.3f}")
        return self

    def predict_for_project(self, project_id, all_subs_list):
        """
        Returns predicted collaborative ratings for all subcontractors for a project.
        Ratings are normalized to [0.0, 1.0] for hybrid blending.
        """
        if not self.is_fitted:
            raise RuntimeError("CollaborativeRecommender is not fitted.")
            
        results = []
        is_known_project = project_id in self.predicted_matrix_df.index
        
        for sub_id in all_subs_list:
            if is_known_project and sub_id in self.predicted_matrix_df.columns:
                pred_rating = float(self.predicted_matrix_df.loc[project_id, sub_id])
            else:
                # Cold-start fallback: Use subcontractor's historical mean or global prior
                pred_rating = float(self.sub_means.get(sub_id, self.global_mean))
                
            # Normalize 1.0-5.0 rating scale into [0.0, 1.0]
            norm_score = (pred_rating - 1.0) / 4.0
            norm_score = float(np.clip(norm_score, 0.0, 1.0))
            
            results.append({
                "subcontractor_id": sub_id,
                "predicted_rating_5star": round(pred_rating, 2),
                "collaborative_score": round(norm_score, 4)
            })
            
        return pd.DataFrame(results)
