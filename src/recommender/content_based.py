"""
Content-Based Recommendation Engine for Subcontractor Matching.
Uses TF-IDF Vectorization over trade scopes, certifications, and project sectors,
combined with feature matching (Location, Safety EMR, Bonding Capacity).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from config import MODELS_DIR, RECOMMENDER_CONFIG

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.subcontractors_df = None
        self.sub_feature_matrix = None
        self.is_fitted = False
        
    def _create_subcontractor_document(self, row):
        """Constructs rich text document representing subcontractor capabilities."""
        doc = f"{row['trade']} {row['trade']} "  # Boost trade weight
        doc += f"{row.get('specialties_str', '')} "
        doc += f"{row.get('certifications_str', '')} "
        doc += f"Location_{row.get('primary_location', '').replace(' ', '_')} "
        if row.get('safety_emr', 1.0) <= 0.85:
            doc += "High_Safety_Tier_1 "
        return doc.strip()
        
    def _create_project_query_document(self, project_row):
        """Constructs text document representing project requirements."""
        doc = f"{project_row['required_trade']} {project_row['required_trade']} "
        doc += f"{project_row.get('sector', '')} "
        doc += f"Location_{project_row.get('location', '').replace(' ', '_')} "
        if project_row.get('bim_lod_required') == 'LOD 400':
            doc += "BIM-LOD400 3D_BIM_Coordination "
        if 'Healthcare' in project_row.get('sector', ''):
            doc += "ASHE-Healthcare Infection_Control "
        if 'Biopharmaceutical' in project_row.get('sector', ''):
            doc += "Cleanroom-Certified "
        return doc.strip()

    def fit(self, subcontractors_df):
        """Fits TF-IDF vectorizer over all registered subcontractors."""
        self.subcontractors_df = subcontractors_df.copy().reset_index(drop=True)
        docs = [self._create_subcontractor_document(row) for _, row in self.subcontractors_df.iterrows()]
        self.sub_feature_matrix = self.vectorizer.fit_transform(docs)
        self.is_fitted = True
        
        # Save fitted model artifact
        joblib.dump(self.vectorizer, MODELS_DIR / "content_vectorizer.pkl")
        return self

    def score_all_for_project(self, project_dict):
        """
        Calculates content-based similarity scores between a given project and all subcontractors.
        Returns a DataFrame of [subcontractor_id, content_score, trade_match, location_match].
        """
        if not self.is_fitted:
            raise RuntimeError("ContentBasedRecommender must be fitted before predicting.")
            
        proj_doc = self._create_project_query_document(project_dict)
        proj_vec = self.vectorizer.transform([proj_doc])
        
        # Base cosine similarity [0.0, 1.0]
        text_sim = cosine_similarity(proj_vec, self.sub_feature_matrix)[0]
        
        scores = []
        for idx, row in self.subcontractors_df.iterrows():
            sim = float(text_sim[idx])
            
            # Feature alignment bonuses and safety penalties
            same_trade = 1.0 if row["trade"].strip().lower() == project_dict["required_trade"].strip().lower() else 0.0
            same_loc = 1.0 if row["primary_location"] == project_dict["location"] else 0.0
            
            # EMR Safety penalty / bonus
            emr = row.get("safety_emr", 1.0)
            if emr <= 0.85:
                safety_modifier = 0.10
            elif emr > RECOMMENDER_CONFIG["max_allowed_emr"]:
                safety_modifier = -0.30
            else:
                safety_modifier = 0.0
                
            # Bonding capacity alignment
            budget = project_dict.get("budget_usd", 10_000_000)
            bonding = row.get("bonding_capacity_usd", 5_000_000)
            bonding_fit = 0.05 if bonding >= (budget * 0.5) else -0.10
            
            # Composite content score scaled to [0.0, 1.0]
            # Primary weight on trade match and cosine similarity
            composite = (0.40 * same_trade) + (0.35 * sim) + (0.10 * same_loc) + (0.10 * (1.0 + safety_modifier) / 2) + (0.05 * (1.0 + bonding_fit) / 2)
            composite = float(np.clip(composite, 0.0, 1.0))
            
            scores.append({
                "subcontractor_id": row["subcontractor_id"],
                "content_score": round(composite, 4),
                "same_trade": bool(same_trade),
                "same_location": bool(same_loc),
                "cosine_sim": round(float(sim), 4)
            })
            
        return pd.DataFrame(scores)
