# Датасет оттока клиентов

**Источник:** генератор `src/data/generate.py`, сид `seed` из `params.yaml`
**Владелец:** Галиев Камиль

**Обновление:** по запросу, вручную — перегенерация + `dvc add` + `dvc push`.
Регулярных поставок нет.

## Поля

| Поле | Тип | Единица | Диапазон | Комментарий |
|---|---|---|---|---|
| customer_id | str | — | `C0000000`… | уникальный идентификатор клиента |
| tenure_months | int | месяцы | 1–72 | стаж клиента |
| monthly_charges | float | руб./мес | 15–180 | текущий тариф; зависит от `internet_service` |
| total_charges | float | руб. | >= 0 | сумма списаний ≈ `monthly_charges × tenure_months`; ~1 % пропусков |
| contract_type | category | — | `month-to-month`, `one_year`, `two_year` | тип договора |
| internet_service | category | — | `fiber`, `dsl`, `none` | тип подключения |
| payment_method | category | — | `electronic_check`, `mailed_check`, `bank_transfer`, `credit_card` | способ оплаты |
| num_support_calls | int | звонки | 0-... | обращения в поддержку |
| has_tech_support | int (0/1) | — | 0, 1 | подключена техподдержка|
| avg_monthly_gb | float | ГБ/мес | 0.5–300 | средний трафик в месяц |
| is_senior | int (0/1) | — | 0, 1 | клиент пенсионного возраста |
| churn | int (0/1) | — | 0, 1 | **таргет**: клиент ушёл |

## Известные дефекты

- `total_charges` пустой у ~1% строк. Пропуски распределены
  произвольно по всем клиентам.
  Можно осстанавливать как `monthly_charges * tenure_months`.
- `monthly_charges` обрезан снизу на 15.0: у ~4% строк ровно 15.0, и все они —
  `internet_service = none`.
- `avg_monthly_gb` обрезан снизу на 0.5 — у клиентов без интернета трафик всё равно ненулевой.
- `total_charges` линейно зависим от `tenure_months * monthly_charges`, сильная коллинеарность.

## Данные
Remote лежит в локальом minio хранилище
```
dvc remote modify --local minio access_key_id <access_key>
dvc remote modify --local minio secret_access_key <access_password>
dvc pull
```

## Почему .dvc надо коммитить

В .dvc хранится конфиг с точками доступа к данным. При настройке проекта с нуля доступен список источников для загрузки данных.