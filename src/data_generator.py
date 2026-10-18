"""
Synthetic Construction Data Generator for ConstructRec.
Simulates real-world enterprise construction data matching DPR Construction / vConstruct domains:
- Projects (Healthcare, Data Centers, Biopharma, Commercial)
- Subcontractors (Trade specialists, Safety EMR ratings, BIM capabilities)
- Historical Performance Reviews (Evaluations, Schedule adherence, Quality scores)
"""
import json
import random
import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from config import RAW_DATA_DIR, TRADE_CATEGORIES, PROJECT_SECTORS, LOCATIONS

random.seed(42)
np.random.seed(42)

def generate_subcontractors(n=120):
    """Generate realistic commercial subcontractors with certifications and safety metrics."""
    prefixes = [
        "Apex", "Pinnacle", "Vanguard", "Precision", "Titan", "Summit", "Sterling", 
        "Dynamic", "Optima", "Benchmark", "Paramount", "Allied", "Omega", "Beacon", "Integra"
    ]
    suffixes = ["Solutions", "Systems", "Contractors", "Engineering", "Builders", "Specialties", "Group"]
    
    cert_pool = ["OSHA-30", "USGBC-LEED", "ISO-9001", "ASHE-Healthcare", "BIM-LOD400", "Cleanroom-Certified"]
    
    subcontractors = []
    for i in range(1, n + 1):
        sub_id = f"SUB-{i:04d}"
        trade = random.choice(TRADE_CATEGORIES)
        name = f"{random.choice(prefixes)} {trade.split()[0]} {random.choice(suffixes)}"
        loc = random.choice(LOCATIONS)
        
        # Safety EMR: industry benchmark is 1.0; <= 0.85 is stellar, > 1.05 has risk
        emr = round(float(np.clip(np.random.normal(0.88, 0.12), 0.60, 1.25)), 2)
        
        # Bonding capacity ($2M to $60M depending on company scale)
        bonding = int(np.random.choice([2_000_000, 5_000_000, 15_000_000, 35_000_000, 60_000_000], 
                                       p=[0.25, 0.35, 0.20, 0.15, 0.05]))
        
        experience = int(np.random.randint(4, 35))
        crew_count = int(np.random.randint(15, 250))
        
        # Pick 1 to 4 certifications
        k_certs = random.choices([1, 2, 3, 4], weights=[0.2, 0.4, 0.3, 0.1])[0]
        certs = sorted(list(set(random.sample(cert_pool, k=k_certs))))
        
        # Specializations tags
        specialties = [trade]
        if "BIM-LOD400" in certs:
            specialties.append("3D BIM Coordination")
        if "ASHE-Healthcare" in certs:
            specialties.append("Infection Control & Medical Gas")
        if "Cleanroom-Certified" in certs:
            specialties.append("ISO Class 5/7 Cleanroom Assembly")
            
        subcontractors.append({
            "subcontractor_id": sub_id,
            "company_name": name,
            "trade": trade,
            "primary_location": loc,
            "safety_emr": emr,
            "bonding_capacity_usd": bonding,
            "experience_years": experience,
            "active_crew_count": crew_count,
            "certifications": certs,
            "specialties": specialties,
            "is_active": True
        })
    return subcontractors

def generate_projects(n=60):
    """Generate commercial and mission-critical construction projects."""
    project_names = [
        "BioNexus Therapeutics Lab", "Skyline Health Pavilion", "HyperScale Data Center Alpha",
        "NextGen Semiconductor Fab", "Centennial Medical Center", "Vertex Innovation Tower",
        "Apex Life Sciences Hub", "Quantum Compute Facility", "Horizon Cancer Research Center",
        "Prism Corporate Campus", "St. Jude Outpatient Wing", "TerraWatt Battery Gigafactory",
        "Metropolitan Surgical Hospital", "Cirrus Cloud Infrastructure Facility", "Olympus Biologics Plant"
    ]
    
    projects = []
    for i in range(1, n + 1):
        proj_id = f"PRJ-{i:04d}"
        base_name = project_names[(i - 1) % len(project_names)]
        name = f"{base_name} Phase {((i - 1) // len(project_names)) + 1}"
        sector = random.choice(PROJECT_SECTORS)
        loc = random.choice(LOCATIONS)
        budget = int(np.random.choice([5_000_000, 18_000_000, 45_000_000, 85_000_000, 150_000_000], 
                                      p=[0.20, 0.35, 0.25, 0.15, 0.05]))
        sqft = int(budget * random.uniform(8.5, 12.0))
        
        # Required trade package for this bidding scope
        req_trade = random.choice(TRADE_CATEGORIES)
        
        # BIM & Quality requirements
        bim_lod = "LOD 400" if sector in ["Healthcare (Hospitals & Clinics)", "Advanced Technology (Data Centers & Semis)", "Biopharmaceutical & Lab Facilities"] else "LOD 300"
        strict_safety = True if sector in ["Advanced Technology (Data Centers & Semis)", "Biopharmaceutical & Lab Facilities"] else False
        
        projects.append({
            "project_id": proj_id,
            "project_name": name,
            "sector": sector,
            "location": loc,
            "budget_usd": budget,
            "sq_footage": sqft,
            "required_trade": req_trade,
            "bim_lod_required": bim_lod,
            "strict_safety_mandate": strict_safety,
            "status": "Planning" if i > 40 else "Completed"
        })
    return projects

def generate_historical_evaluations(subcontractors, projects, n_interactions=750):
    """
    Generate historical Project-Manager rating records on completed projects.
    High similarity in trade + high quality yields realistic clustered ratings.
    """
    completed_projects = [p for p in projects if p["status"] == "Completed"]
    evaluations = []
    
    trade_map = {}
    for sub in subcontractors:
        trade_map.setdefault(sub["trade"], []).append(sub)
        
    for eval_id in range(1, n_interactions + 1):
        proj = random.choice(completed_projects)
        trade = proj["required_trade"]
        
        # 85% of time matched by trade, 15% cross-discipline bid
        if trade in trade_map and random.random() < 0.85:
            sub = random.choice(trade_map[trade])
        else:
            sub = random.choice(subcontractors)
            
        # Ground truth rating calculation based on safety, location proximity, and inherent capability
        base_rating = 4.2 if sub["trade"] == proj["required_trade"] else 2.5
        
        # EMR impact (lower is safer/better)
        if sub["safety_emr"] <= 0.85:
            base_rating += 0.4
        elif sub["safety_emr"] > 1.05:
            base_rating -= 0.8
            
        # Location proximity boost
        if sub["primary_location"] == proj["location"]:
            base_rating += 0.3
            
        # Noise
        final_rating = round(float(np.clip(base_rating + np.random.normal(0, 0.35), 1.0, 5.0)), 1)
        schedule_score = round(float(np.clip(final_rating + np.random.normal(0, 0.3), 1.0, 5.0)), 1)
        safety_score = round(float(np.clip((1.2 - sub["safety_emr"]) * 4.0 + 1.0, 1.0, 5.0)), 1)
        
        evaluations.append({
            "evaluation_id": f"EVL-{eval_id:05d}",
            "project_id": proj["project_id"],
            "subcontractor_id": sub["subcontractor_id"],
            "rating": final_rating,
            "schedule_score": schedule_score,
            "safety_score": safety_score,
            "change_order_rate": round(float(random.uniform(0.01, 0.12)), 3),
            "recommendation_confirmed": True if final_rating >= 3.8 else False,
            "timestamp": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })
        
    return evaluations

def generate_all_raw_data():
    """Generates and writes all Bronze layer raw JSON files."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[DataGen] Generating synthetic enterprise datasets...")
    subcontractors = generate_subcontractors(n=120)
    projects = generate_projects(n=60)
    evaluations = generate_historical_evaluations(subcontractors, projects, n_interactions=800)
    
    sub_file = RAW_DATA_DIR / "subcontractors_raw.json"
    prj_file = RAW_DATA_DIR / "projects_raw.json"
    evl_file = RAW_DATA_DIR / "evaluations_raw.json"
    
    with open(sub_file, "w") as f:
        json.dump(subcontractors, f, indent=2)
    with open(prj_file, "w") as f:
        json.dump(projects, f, indent=2)
    with open(evl_file, "w") as f:
        json.dump(evaluations, f, indent=2)
        
    print(f"[DataGen] Ingested {len(subcontractors)} subcontractors -> {sub_file.name}")
    print(f"[DataGen] Ingested {len(projects)} projects -> {prj_file.name}")
    print(f"[DataGen] Ingested {len(evaluations)} historical evaluations -> {evl_file.name}")
    return sub_file, prj_file, evl_file

if __name__ == "__main__":
    generate_all_raw_data()
