# Resultado final — V3 (multi-stage + cohort-aware + stratified calibration)

## Métricas globais (175k VINs)

- **n**: 175552
- **churn_real**: 0.4898
- **accuracy**: 0.8701
- **balanced_accuracy**: 0.8697
- **precision**: 0.8812
- **recall**: 0.8493
- **f1**: 0.8649
- **roc_auc**: 0.9262
- **pr_auc**: 0.9304
- **brier**: 0.1275

## Métricas por scoring_quality

| Quality | n | acc | balanced_acc | F1 | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| high_confidence | 116,632 | 0.834 | 0.834 | 0.832 | 0.907 | 0.167 |
| holdout | 31,938 | 0.900 | 0.889 | 0.864 | 0.908 | 0.086 |
| single_event | 26,982 | 0.992 | 0.993 | 0.993 | 0.998 | 0.007 |

## Métricas por cohort de venda

| Ano | n | churn_real | score | ROC-AUC | balanced_acc |
|---:|---:|---:|---:|---:|---:|
| 2020 | 64,634 | 0.927 | 0.680 | 0.9979 | 0.957 |
| 2021 | 18,960 | 0.757 | 0.456 | 0.9981 | 0.902 |
| 2022 | 12,137 | 0.432 | 0.177 | 0.9936 | 0.841 |
| 2023 | 21,094 | 0.214 | 0.144 | 0.9026 | 0.748 |
| 2024 | 37,029 | 0.052 | 0.183 | 0.6555 | 0.552 |
| 2025 | 21,568 | 0.000 | 0.180 | 0.3098 | 0.451 |
| 2026 | 130 | 0.008 | 0.173 | 0.9961 | 0.996 |

## Resumo dos erros (top 5 modelos com pior acc)

| Modelo | n | Acc | %FP | %FN |
|---|---:|---:|---:|---:|
| TRANSIT | 4,120 | 0.788 | 0.087 | 0.125 |
| TERRITORY | 11,376 | 0.801 | 0.135 | 0.065 |
| RANGER | 78,225 | 0.809 | 0.092 | 0.100 |
| F-150 | 1,681 | 0.857 | 0.113 | 0.030 |
| BRONCO SPORT | 5,689 | 0.872 | 0.038 | 0.090 |