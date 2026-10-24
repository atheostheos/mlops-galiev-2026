# Топ 5 моделей из MLFlow по ROC-AUC

| | model | params | ROC-AUC | PR-AUC | git_sha | date | run_id |
|---|---|---|---|---|---|---|---|
| текущая | logreg | max_iter=1000, C=0.5 | 0.8086 | 0.5193 | f7b670f | 2026-10-24 12:29 | 9d56fa28 |
|  | logreg | max_iter=1000, C=1.0 | 0.8086 | 0.5193 | f7b670f | 2026-10-24 12:28 | defce216 |
|  | logreg | max_iter=1000, C=2.0 | 0.8086 | 0.5193 | f7b670f | 2026-10-24 12:30 | e5adb518 |
|  | gradient_boosting | learning_rate=0.05, n_estimators=200, max_depth=3 | 0.8044 | 0.5017 | f7b670f | 2026-10-24 12:26 | c85dbc22 |
|  | random_forest | min_samples_leaf=20, max_depth=20, n_estimators=300 | 0.8019 | 0.4974 | f7b670f | 2026-10-24 12:32 | deec3af8 |
