# Comparison of Machine Learning Models: Custom Implementations vs scikit-learn (C++ and Python)
## EN :uk: / RU :ru:
**EN :uk:**

A project for experimental comparison of classic machine learning models implemented in three variants:

1. **`scikit-learn` library** (baseline);
2. **Custom implementation in Python**;
3. **Custom implementation in C++**.

The comparison is based on two main criteria:

- **model quality** (error / accuracy on the test data);
- **performance** (training and inference time).

At the moment, **linear regression** is implemented and compared; other models are under development.

---

## Project goals

- Gain practical understanding of how close one can get to `scikit-learn` by implementing models “from scratch”.
- Explore the difference between Python and C++ when implementing the same algorithms:
  - language and runtime overhead;
  - development convenience;
  - code maintainability.
- Build reproducible experiments that allow you to:
  - run identical training and testing scenarios;
  - log quality metrics;
  - perform fair time benchmarks.

---

## Repository architecture and structure

```text
from-scratch-py-vs-cpp-ml/
├── data/                 # Source data for experiments
│   └── regression/       # Datasets for regression tasks
├── notebooks/            # Jupyter notebooks with experiments and visualizations
├── src/                  # Algorithm implementations in C++ and Python
├── vendor/               # External / helper code (headers, utilities, etc.)
├── requirements.txt      # Python dependencies
└── .gitmodules           # Git submodules configuration
```

---

### Metrics and preprocessing implementation

Within the project we **deliberately implement by hand** both the main metrics and data preprocessing:

**Metrics (Python and C++):**

- R²  
- D²  
- RMSE  
- SMAPE  
- RMSLE  
- MAXE
- MedAE
- EVS

**Preprocessing:**

- `StandardScaler` — custom implementation of feature standardization (estimating mean and standard deviation on the train part and applying it to train/test);
- `train_test_split` — custom implementation of splitting the dataset into train / test with a fixed random seed and shuffle.

The goals of this approach are:

- better understand numerical behavior and possible sources of errors;
- compare not only the models, but also the cost of computing metrics / transformations in Python and C++;

At the same time, **we intentionally use `scikit-learn` only as an analysis utility**:

- **`learning_curve`**  
- **`permutation_test_score`**

These functions are taken from `sklearn.model_selection` and used for:

- building learning curves and analyzing generalization ability;
- assessing the statistical significance of model quality.

All experimental notebooks explicitly indicate which parts of the pipeline are implemented manually and which parts use `scikit-learn` as a helper tool.

---

### Linear regression

Evaluation is performed **on the same dataset** (for now we use a single dataset; later we plan to add several heterogeneous datasets) using the following set of metrics:

- **R²** — coefficient of determination;
- **D²** — a modification of R² (deviance-based score) for assessing prediction quality;
- **RMSE** — a variation of mean squared error (mean root square error);
- **SMAPE** — symmetric mean absolute percentage error;
- **RMSLE** — root mean squared logarithmic error;
- **MAXE** — maximum absolute error (the worst mistake the model makes on a single instance);
- **MedAE** — median absolute error (a typical error, robust to outliers);
- **EVS** — proportion of the target variable’s variance explained by the model (how well the model explains the target’s variation).

In addition to point metrics, we use tools to analyze generalization and statistical significance:

- **`learning_curve`** — to build learning curves and analyze how quality depends on the size of the training set;
- **`permutation_test_score`** — to assess the statistical significance of model quality and check whether the result could be due to chance.

---

## **RU :ru:**

# Сравнение моделей машинного обучения: свои реализации vs scikit-learn (C++ и Python)

Проект для экспериментального сравнения реализаций классических моделей машинного обучения в трёх вариантах:

1. **Библиотека `scikit-learn`** (baseline);
2. **Собственная реализация на Python**;
3. **Собственная реализация на C++**.

Сравнение ведётся по двум основным критериям:

- **качество модели** (ошибка / точность на тестовых данных);
- **производительность** (время обучения и предсказания).

В настоящий момент реализована и сравнивается **линейная регрессия**; остальные модели находятся в разработке.

---

## Цели проекта

- Получить практическое понимание того, насколько близко к `scikit-learn` можно подойти, реализуя модели «вручную».
- Исследовать разницу между Python и C++ при реализации одних и тех же алгоритмов:
  - накладные расходы языка и рантайма;
  - удобство разработки;
  - сложность поддержки кода.
- Построить воспроизводимые эксперименты, в которых можно:
  - запускать одинаковые сценарии обучения и тестирования;
  - фиксировать метрики качества;
  - проводить честные бенчмарки по времени.

---

## Архитектура и структура репозитория

```text
from-scratch-py-vs-cpp-ml/
├── data/                 # Исходные данные для экспериментов
│   └── regression/       # Датасеты для задач регрессии
├── notebooks/            # Jupyter-ноутбуки с экспериментами и визуализацией
├── src/                  # Реализации алгоритмов на C++ и Python
├── vendor/               # Внешний/вспомогательный код (заголовки, утилиты и пр.)
├── requirements.txt      # Python-зависимости
└── .gitmodules           # Конфигурация git-подмодулей
```

---

### Реализация метрик и предобработки

В рамках проекта мы **осознанно реализуем вручную** как основные метрики, так и предобработку данных:

**Метрики (Python и C++):**

- R²  
- D²  
- RMSE  
- SMAPE  
- RMSLE
- MAXE
- MedAE
- EVS

**Предобработка:**

- `StandardScaler` — собственная реализация стандартизации признаков (оценка среднего и стандартного отклонения по train-части, применение к train/test).
- `train_test_split` — собственная реализация разбиения выборки на train / test с фиксированным random seed и shuffle.


Цели такого подхода:

- лучше понимать численное поведение и возможные источники ошибок;
- сравнивать не только модели, но и стоимость вычисления метрик/преобразований в Python и C++;

При этом мы **сознательно используем `scikit-learn` только как утилиту анализа**:

- **`learning_curve`**  
- **`permutation_test_score`**

Эти функции берутся из `sklearn.model_selection` и используются для:

- построения кривых обучения и анализа обобщающей способности;
- оценки статистической значимости качества моделей.

Во всех экспериментальных ноутбуках явно указано, какие части пайплайна реализованы вручную, а какие используют `sklearn` как вспомогательный инструмент.

---

### Линейная регрессия

Оценка проводится **на одном и том же датасете** (пока что используем один датасет; в дальнейшем планируем добавить несколько разнородных наборов данных) по следующему набору метрик:

- **R²** — коэффициент детерминации;
- **D²** — модификация R² (deviance-based score) для оценки качества предсказаний;
- **RMSE** — вариация средней квадратичной ошибки (mean root square error);
- **SMAPE** — symmetric mean absolute percentage error;
- **RMSLE** — root mean squared logarithmic error;
- **MAXE** — максимальная абсолютная ошибка (наихудший промах модели по одному объекту);
- **MedAE** — медианная абсолютная ошибка (типичная ошибка, устойчива к выбросам);
- **EVS** — доля дисперсии целевой переменной, объяснённая моделью (насколько хорошо модель объясняет вариацию таргета).

Помимо точечных метрик, мы используем инструменты анализа обобщающей способности и статистической значимости:

- **`learning_curve`** — для построения кривых обучения и анализа зависимости качества от размера обучающей выборки;
- **`permutation_test_score`** — для оценки статистической значимости качества модели и проверки, не является ли результат следствием случайности.

