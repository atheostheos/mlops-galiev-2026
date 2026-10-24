# Лабораторная работа №1

Threshold = 0.5
- Logreg:
    ```json
    {
        "roc_auc": 0.8085810447286689,
        "pr_auc": 0.519275321207295,
        "f1": 0.38752052545155996
    }
    ```
- Random Forest:
    ```json
    {
        "roc_auc": 0.7738446418527304,
        "pr_auc": 0.46705632478932996,
        "f1": 0.3791469194312796
    }
    ```
- Gradient Boosting:
    ```json
    {
        "roc_auc": 0.8041248379839085,
        "pr_auc": 0.5044991693606045,
        "f1": 0.3723228995057661
    }
    ```

ROC-AUC считает без учета выбора порога и измеряет способность модели определять разницу между наличием churn-a и его отсуствием.

F1 высчитывается по выбранному порогу. Технически низкий f1 может сказать о том, что нужно изменить threshold.

В итоге: Высокий ROC-AUC говорит о хорошей способности модели интепретировать данные, но threshold нужно изменить достижения для лучшей точности.

## Лабораторная работа №3

| № | Модель |  Гиперпараметры | ROC-AUC | PR-AUC | F1 |
|---|--------|-----------------|---------|--------|----|
| 1 | logreg | `C=1.0, max_iter=1000` | 0.8086 | 0.5193 | 0.3875 |
| 2 | logreg | `C=0.05, max_iter=1000` | 0.8088 | 0.5203 | 0.3808 |
| 3 | random_forest | `n_estimators=300, max_depth=12, min_samples_leaf=5` | 0.7982 | 0.4925 | 0.3652 |
| 4 | random_forest | `n_estimators=500, max_depth=6, min_samples_leaf=20` | 0.8030 | 0.4959 | 0.3099 |
| 5 | gradient_boosting | `n_estimators=200, learning_rate=0.05, max_depth=3` | 0.8044 | 0.5017 | 0.3738 |
| 6 | gradient_boosting | `n_estimators=400, learning_rate=0.1, max_depth=5` | 0.7747 | 0.4677 | 0.4122 |

Ручная запись результатов требует много времени + есть шанс некорректно записать результаты. Можно было бы автоматизировать подбор гиперпараметров одним запуском и добавить формирование отчета.

## DVC METRICS

Проверка метрик с коммита `77012cc third experiment`

```bash
>> dvc metrics diff HEAD~2  

Path                        Metric    HEAD~2    workspace    Change
reports/train_metrics.json  f1        0.38752   0.37377      -0.01375
reports/train_metrics.json  pr_auc    0.51928   0.50166      -0.01761
reports/train_metrics.json  roc_auc   0.80858   0.80438      -0.0042
reports/eval_metrics.json   f1        0.41868   0.39676      -0.02192
reports/eval_metrics.json   pr_auc    0.52883   0.51453      -0.0143
reports/eval_metrics.json   roc_auc   0.80744   0.79944      -0.008
```

```bash
>> dvc metrics diff HEAD~1

Path                        Metric    HEAD~1    workspace    Change
reports/train_metrics.json  f1        0.36516   0.37377      0.00861
reports/train_metrics.json  pr_auc    0.49248   0.50166      0.00918
reports/train_metrics.json  roc_auc   0.79821   0.80438      0.00617
reports/eval_metrics.json   f1        0.38736   0.39676      0.0094
reports/eval_metrics.json   pr_auc    0.50667   0.51453      0.00786
reports/eval_metrics.json   roc_auc   0.79552   0.79944      0.00393
```

```bash
>> dvc metrics diff HEAD~2 HEAD~1

Path                        Metric    HEAD~2    HEAD~1    Change
reports/train_metrics.json  f1        0.38752   0.36516   -0.02236
reports/train_metrics.json  pr_auc    0.51928   0.49248   -0.02679
reports/train_metrics.json  roc_auc   0.80858   0.79821   -0.01037
reports/eval_metrics.json   f1        0.41868   0.38736   -0.03131
reports/eval_metrics.json   pr_auc    0.52883   0.50667   -0.02215
reports/eval_metrics.json   roc_auc   0.80744   0.79552   -0.01193
```
