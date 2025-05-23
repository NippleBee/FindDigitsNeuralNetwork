# Примеры использования нейросети для распознавания капч

В этом документе приведены примеры использования нейросети для распознавания 5-значных цифровых капч.

## 1. Генерация датасета

Для генерации датасета из 10 000 изображений капч выполните команду:

```bash
python src/data_generator.py --num_samples 10000 --output_dir dataset
```

Параметры:

- `--num_samples` - количество изображений для генерации
- `--output_dir` - директория для сохранения датасета

## 2. Обучение модели

Для обучения модели на сгенерированном датасете выполните команду:

```bash
python src/train.py --dataset_dir dataset --model_dir models --epochs 50 --batch_size 32 --augmentation --evaluate
```

Параметры:

- `--dataset_dir` - директория с датасетом
- `--model_dir` - директория для сохранения моделей
- `--epochs` - количество эпох обучения
- `--batch_size` - размер батча
- `--augmentation` - применять аугментации к обучающему датасету
- `--evaluate` - оценить модель на тестовом датасете после обучения

## 3. Оценка модели

Для оценки обученной модели на тестовом датасете выполните команду:

```bash
python src/inference.py --mode evaluate --model_path models/best_model.keras --test_dir dataset/test --output_dir results
```

Параметры:

- `--mode` - режим работы (evaluate - оценка на тестовом датасете)
- `--model_path` - путь к файлу модели
- `--test_dir` - директория с тестовыми данными
- `--output_dir` - директория для сохранения результатов

## 4. Распознавание одного изображения

Для распознавания одного изображения капчи выполните команду:

```bash
python src/inference.py --mode predict --model_path models/best_model.keras --image_path path/to/captcha.png
```

Параметры:

- `--mode` - режим работы (predict - распознавание одного изображения)
- `--model_path` - путь к файлу модели
- `--image_path` - путь к изображению для распознавания

## 5. Запуск полного пайплайна

Для запуска полного пайплайна (генерация датасета, обучение, оценка) выполните команду:

```bash
python src/run_pipeline.py --run_all --num_samples 10000 --epochs 50 --augmentation
```

Параметры:

- `--run_all` - запустить весь пайплайн
- `--num_samples` - количество изображений для генерации
- `--epochs` - количество эпох обучения
- `--augmentation` - применять аугментации к обучающему датасету

## 6. Советы по подготовке датасета

1. **Размер датасета**: Для достижения высокой точности рекомендуется использовать не менее 10 000 изображений.

2. **Разнообразие данных**: Убедитесь, что в датасете представлены различные вариации цифр, фона и шумов.

3. **Аугментации**: Используйте аугментации для увеличения разнообразия данных и улучшения обобщающей способности модели.

4. **Баланс классов**: Убедитесь, что все цифры (0-9) представлены примерно в равных пропорциях.

5. **Качество изображений**: Генерируйте изображения с параметрами, близкими к реальным капчам, которые нужно распознавать.

## 7. Оптимизация модели

Если требуется улучшить скорость инференса или точность модели, попробуйте следующие подходы:

1. **Уменьшение размера модели**: Уменьшите количество слоев или фильтров в сверточных слоях.

2. **Квантизация**: Примените квантизацию для уменьшения размера модели и ускорения инференса.

3. **Pruning**: Удалите неважные веса из модели для уменьшения её размера.

4. **Дистилляция знаний**: Обучите меньшую модель на основе предсказаний большой модели.

5. **Оптимизация гиперпараметров**: Подберите оптимальные значения гиперпараметров (learning rate, batch size, архитектура сети).
