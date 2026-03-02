# Face Recognition Contest Project (Windows + PyCharm)

## 1) Краткий анализ задания
- Нужен локальный демонстрационный ML-проект для распознавания лиц без Docker.
- Требуются: аудит данных из `Data1.zip..Data4.zip`, обучение модели, инференс, API (FastAPI), GUI (Tkinter), notebooks и тесты.
- Решение должно быть практичным для защиты: быстрый запуск, понятная структура, отчеты и метрики.

## 2) Выбранный подход
- Детекция лица: OpenCV Haar Cascade (быстро и просто для офлайн-демо).
- Классификация личности: компактная CNN на PyTorch.
- Pipeline:
  1. Аудит + распаковка архивов
  2. Кроп лиц + сбор `train/val`
  3. Обучение + сохранение `models/face_classifier.pt`
  4. Оценка метрик
  5. API + GUI для демонстрации

## 3) Архитектура
- `src/data`: аудит данных и предобработка.
- `src/models`: dataset, модель, обучение, инференс, evaluation.
- `src/api`: FastAPI endpoints.
- `src/gui`: Tkinter desktop app.
- `src/utils`: конфиг и логирование.
- `notebooks`: аналитика/демонстрация.

## 4) Структура проекта
```text
.
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── gui/
│   ├── models/
│   └── utils/
└── tests/
```

## 5) Зависимости
См. `requirements.txt`.
Основные: `torch`, `opencv-python`, `fastapi`, `uvicorn`, `pytest`, `jupyter`, `pandas`, `scikit-learn`, `Pillow`.

## 6) План реализации
1. Положить архивы `Data1.zip..Data4.zip` в `data/raw/`.
2. Запустить аудит данных и предобработку.
3. Обучить модель и проверить метрики.
4. Поднять API и проверить `/predict`.
5. Запустить Tkinter GUI для демонстрации.
6. Прогнать тесты `pytest`.

---

## Быстрый старт в PyCharm (Windows)

### 1. Создайте окружение Python 3.10+
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Установите PyTorch (если нужно выбрать CPU/CUDA)
Пример CPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Подготовка данных
- Скопируйте архивы в `data/raw/`:
  - `Data1.zip`
  - `Data2.zip`
  - `Data3.zip`
  - `Data4.zip`

### 4. Запуск полного пайплайна
```bash
python -m src.main
```

### 5. Запуск API
```bash
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```
Swagger: http://127.0.0.1:8000/docs

### 6. Запуск GUI
```bash
python -m src.gui.app
```

### 7. Тесты
```bash
pytest -q
```

---

## Endpoints
- `GET /health`
- `GET /model-info`
- `POST /predict` (multipart image file)

Ответ `predict` включает:
- `bbox`
- `class`
- `confidence`
- `annotated_image`

## Отчеты
- `reports/data_audit_report.csv`
- `reports/data_issues.json`
- `reports/preprocessing_report.csv`
- `reports/training_history.json`
- `reports/evaluation_metrics.json`

