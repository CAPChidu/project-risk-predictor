"""Project Data Generator
Generates realistic historical project data for training the ML model.
Simulates real-world project scenarios with various risk factors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_project_data(n_projects=500):
    """
    Generate synthetic project data with realistic correlations.
    
    Features include:
    - Project characteristics (budget, team size, duration)
    - Risk factors (scope creep, resource availability, stakeholder engagement)
    - Outcome (success/failure)
    """
    
    data = []
    
    project_types = ['Software Development', 'Infrastructure', 'Business Process', 
                    'Product Launch', 'System Integration']
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
    
    for i in range(n_projects):
        # Basic project attributes
        project_type = random.choice(project_types)
        industry = random.choice(industries)
        
        # Budget (in thousands)
        budget = round(np.random.lognormal(mean=5, sigma=1) * 10, 2)
        budget = max(50, min(budget, 5000))  # Between 50k and 5M
        
        # Team size
        team_size = int(np.random.lognormal(mean=2, sigma=0.8))
        team_size = max(3, min(team_size, 50))
        
        # Project duration (months)
        planned_duration = int(np.random.gamma(shape=3, scale=2))
        planned_duration = max(1, min(planned_duration, 24))
        
        # Risk factors (1-10 scale)
        scope_clarity = round(np.random.beta(a=5, b=2) * 10, 1)
        stakeholder_engagement = round(np.random.beta(a=4, b=2) * 10, 1)
        resource_availability = round(np.random.beta(a=5, b=3) * 10, 1)
        technical_complexity = round(np.random.uniform(1, 10), 1)
        team_experience = round(np.random.beta(a=5, b=2) * 10, 1)
        
        # External factors
        change_requests = int(np.random.poisson(lam=3))
        dependencies = int(np.random.poisson(lam=2))
        
        # Risk score (normalized)
        risk_score = (
            (10 - scope_clarity) * 0.2 +
            (10 - stakeholder_engagement) * 0.15 +
            (10 - resource_availability) * 0.2 +
            technical_complexity * 0.15 +
            (10 - team_experience) * 0.15 +
            (change_requests / 10) * 0.1 +
            (dependencies / 10) * 0.05
        )
        
        # Success determination (higher risk = lower success probability)
        success_probability = 1 / (1 + np.exp((risk_score - 5) / 2))  # Sigmoid
        success = 1 if random.random() < success_probability else 0
        
        # Actual outcomes (if failed, values are worse)
        if success:
            actual_duration = planned_duration + int(np.random.normal(0, 2))
            budget_variance = round(np.random.normal(0, 15), 1)  # +-15%
            quality_score = round(np.random.beta(a=8, b=2) * 100, 1)
        else:
            actual_duration = planned_duration + int(np.random.normal(5, 3))
            budget_variance = round(np.random.normal(25, 15), 1)  # Overbudget
            quality_score = round(np.random.beta(a=3, b=5) * 100, 1)
        
        actual_duration = max(1, actual_duration)
        
        data.append({
            'project_id': f'PRJ_{i+1:04d}',
            'project_type': project_type,
            'industry': industry,
            'budget_thousands': budget,
            'team_size': team_size,
            'planned_duration_months': planned_duration,
            'actual_duration_months': actual_duration,
            'scope_clarity_score': scope_clarity,
            'stakeholder_engagement': stakeholder_engagement,
            'resource_availability': resource_availability,
            'technical_complexity': technical_complexity,
            'team_experience_level': team_experience,
            'change_requests': change_requests,
            'external_dependencies': dependencies,
            'budget_variance_percent': budget_variance,
            'quality_score': quality_score,
            'success': success
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate dataset
    print("Generating project data...")
    df = generate_project_data(n_projects=500)
    
    # Save to CSV
    df.to_csv('project_data.csv', index=False)
    
    print(f"✓ Generated {len(df)} projects")
    print(f"\nSuccess Rate: {df['success'].mean()*100:.1f}%")
    print(f"\nSample projects:")
    print(df.head())
    print(f"\nFeature Statistics:")
    print(df.describe())
