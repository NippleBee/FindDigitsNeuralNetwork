# Нейросеть для распознавания 5-значных цифровых капч

Данный проект реализует нейронную сеть для распознавания 5-значных цифровых капч с заданными параметрами цветов фона и цифр.

## Структура проекта

```
├── dataset/               # Директория для хранения датасета
│   ├── train/            # Обучающая выборка (80%)
│   ├── validation/       # Валидационная выборка (10%)
│   └── test/             # Тестовая выборка (10%)
├── src/                  # Исходный код
│   ├── data_generator.py # Генерация датасета
│   ├── preprocessing.py  # Предобработка изображений
│   ├── model.py          # Архитектура нейросети
│   ├── train.py          # Обучение модели
│   ├── evaluate.py       # Оценка качества модели
│   └── inference.py      # Инференс модели
├── models/               # Сохраненные модели
├── requirements.txt      # Зависимости проекта
└── README.md             # Документация проекта
```

## Установка и настройка

1. Клонировать репозиторий
2. Установить зависимости:
   ```
   pip install -r requirements.txt
   ```

## Генерация датасета

Для генерации датасета используйте скрипт `data_generator.py`:

```
python src/data_generator.py --num_samples 10000 --output_dir dataset
```

Скрипт создаст 10,000 изображений капч с 5 цифрами, разделит их на обучающую, валидационную и тестовую выборки, и сохранит метки в соответствующих файлах.

## Обучение модели

```
python src/train.py --dataset_dir dataset --model_dir models
```

## Оценка модели

```
python src/evaluate.py --model_path models/best_model.keras --test_dir dataset/test
```

## Инференс

```
python src/inference.py --model_path models/best_model.keras --image_path path/to/captcha.png
```
