# Executive Summary

## Context

This project extends the original Datarisk Junior Data Scientist technical challenge submission into a portfolio-ready decision-support application for delinquency risk prioritization.

## Approach

The workflow combines data preparation, feature engineering, temporal validation, benchmark comparison across Logistic Regression, Random Forest, and LightGBM, threshold tuning, and a bilingual Streamlit interface for executive and technical audiences.
The submission flow preserves one scored output for each record in `base_pagamentos_teste.csv`, and training labels follow the original case rule: delinquent = payment delay `>= 5` days.

## Final Result

- Official final model: `Random Forest`
- Variant: `No SMOTE`
- Official run ID: `56e3d7b850c24eb78a05b2a232f30fd5`
- Selection metric: `PR-AUC`
- PR-AUC: `0.7308`
- ROC-AUC: `0.9598`
- Recommended threshold: `0.20`

## Operational Decision

At the recommended threshold, the model flags about 7.8% of cases, with precision `0.6352` and recall `0.7411`. In practice, this supports a focused collections or risk-review queue that prioritizes the highest-risk cases without overwhelming operational capacity.

## Limitations

- Results reflect the challenge dataset and its specific temporal window.
- Feature importance does not imply causality.
- Threshold policy should be recalibrated to each business context and team capacity.
- Production deployment would require monitoring, governance, and drift controls.
