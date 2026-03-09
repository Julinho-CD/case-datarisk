# Model Card - Delinquency Risk Scoring

## Model Purpose

Predict the probability of delinquency to support operational prioritization in collections and risk analysis workflows.

This model is a decision-support tool, not an autonomous approval/rejection system.

## Challenge Context

Built from the Datarisk Junior Data Scientist technical challenge and later extended as a portfolio project.

## Target Definition

- Target column: `PROBABILIDADE_INADIMPLENCIA`
- Binary label rule (training): delinquent = `1` if payment delay is `>= 5` days, else `0`

## Data Used

- Customer profile data (`base_cadastral.csv`)
- Monthly financial/context data (`base_info.csv`)
- Historical payment behavior (`base_pagamentos_desenvolvimento.csv`)
- Scoring set (`base_pagamentos_teste.csv`)

Datasets are not redistributed in this repository.

## Main Feature Groups

- Payment and invoice attributes (`VALOR_A_PAGAR`, `TAXA`)
- Customer profile (`PORTE`, `SEGMENTO_INDUSTRIAL`, `DOMINIO_EMAIL`, `CEP_2_DIG`)
- Engineered temporal features (`TEMPO_CADASTRO`, `PRAZO_EMISSAO_VENCIMENTO`, `MES`)
- Historical behavior features (`QTDE_ATRASOS_ANT`, `TICKET_MEDIO_ANT`)
- Affordability proxy (`VALOR_RELATIVO_RENDA`)

## Model Selection

Benchmark evaluated Logistic Regression, Random Forest, and LightGBM variants.

Selected best run (by PR-AUC):

- Model: Random Forest (no SMOTE)
- Run ID: `56e3d7b850c24eb78a05b2a232f30fd5`

## Key Metrics (Validation)

- PR-AUC: `0.7308`
- ROC-AUC: `0.9598`
- F1 @ 0.50: `0.5629`
- Best threshold: `0.20`
- F1 @ best threshold: `0.6841`
- Precision @ best threshold: `0.6352`
- Recall @ best threshold: `0.7411`
- Positive rate @ best threshold: `0.0777`

## Recommended Operational Use

1. Score all candidate invoices/customers.
2. Apply threshold policy (`~0.20` baseline recommendation).
3. Prioritize top-risk queue for manual/assisted action.
4. Recalibrate threshold according to team capacity and cost of false negatives.

## Error Trade-offs

- False Negative (high risk missed): delayed collections and potential financial loss.
- False Positive (low risk flagged): unnecessary operational effort.

Threshold choice should explicitly balance these costs.

## Limitations

- Trained on challenge data with specific temporal windows.
- Performance may degrade under distribution shift.
- Feature importance is associative, not causal.
- Missing local model binaries in this repo by design (MLflow-first artifact strategy).

## Risks and Cautions

- Do not use score as sole criterion for irreversible business decisions.
- Monitor fairness and segment-level performance before production deployment.
- Track calibration drift and retrain periodically.

## When to Use

- Prioritization queues in collections/risk operations.
- Workload planning under limited analyst capacity.

## When Not to Use

- Fully automated punitive actions without human review.
- Contexts where legal/compliance constraints require additional explainability controls.
