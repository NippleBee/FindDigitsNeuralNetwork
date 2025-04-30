#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time

# Импортируем наши модули
from preprocessing import CaptchaPreprocessor
from model import CaptchaModel


def train_model(args):
    """Обучает модель для распознавания капч
    
    Args:
        args: Аргументы командной строки
    """
    print("Начинаем обучение модели...")
    
    # Проверяем наличие датасета
    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'validation')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Датасет не найден в {args.dataset_dir}. Генерируем новый датасет...")
        
        # Импортируем генератор датасета
        from data_generator import create_dataset
        
        # Создаем датасет
        create_dataset(
            num_samples=args.num_samples,
            output_dir=args.dataset_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    
    # Создаем препроцессор
    preprocessor = CaptchaPreprocessor(grayscale=args.grayscale)
    
    # Загружаем данные
    print("Загружаем обучающий датасет...")
    train_dataset = preprocessor.load_data(
        train_dir,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Применяем аугментации к обучающему датасету
    if args.augmentation:
        print("Применяем аугментации...")
        train_dataset = preprocessor.apply_augmentation(train_dataset, augmentation_factor=args.augmentation_factor)
    
    print("Загружаем валидационный датасет...")
    validation_dataset = preprocessor.load_data(
        val_dir,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Создаем модель
    model = CaptchaModel(grayscale=args.grayscale)
    
    # Если указан путь к существующей модели, загружаем её
    if args.load_model and os.path.exists(args.load_model):
        print(f"Загружаем существующую модель из {args.load_model}...")
        model.load_model(args.load_model)
    else:
        print("Создаем новую модель...")
        model.build_model()
    
    # Создаем директорию для логов TensorBoard
    log_dir = os.path.join(args.model_dir, 'logs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Добавляем колбэк TensorBoard
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Обучаем модель
    print(f"Начинаем обучение на {args.epochs} эпохах...")
    history = model.train(
        train_dataset,
        validation_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir
    )
    
    # Сохраняем модель
    model_path = os.path.join(args.model_dir, 'final_model.h5')
    model.save_model(model_path)
    
    # Визуализируем историю обучения
    plt.figure(figsize=(15, 5))
    
    # График функции потерь
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Обучение')
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    # График точности для каждой цифры
    plt.subplot(1, 3, 2)
    for i in range(5):
        plt.plot(history.history[f'digit_{i}_accuracy'], label=f'Цифра {i+1}')
    plt.title('Точность по цифрам (обучение)')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    # График средней точности
    plt.subplot(1, 3, 3)
    # Вычисляем среднюю точность по всем выходам
    train_acc = np.mean([history.history[f'digit_{i}_accuracy'] for i in range(5)], axis=0)
    val_acc = np.mean([history.history[f'val_digit_{i}_accuracy'] for i in range(5)], axis=0)
    
    plt.plot(train_acc, label='Обучение')
    plt.plot(val_acc, label='Валидация')
    plt.title('Средняя точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'training_history.png'))
    
    print(f"Обучение завершено. Модель сохранена в {model_path}")
    print(f"График истории обучения сохранен в {os.path.join(args.model_dir, 'training_history.png')}")
    
    # Оцениваем модель на тестовом датасете
    if args.evaluate:
        print("\nОцениваем модель на тестовом датасете...")
        
        # Загружаем тестовый датасет
        test_dir = os.path.join(args.dataset_dir, 'test')
        test_dataset = preprocessor.load_data(
            test_dir,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # Оцениваем модель
        loss, accuracy, overall_accuracy = model.evaluate(test_dataset)
        
        print("\nРезультаты оценки:")
        print(f"Общая точность (все 5 цифр): {overall_accuracy:.4f}")
        
        for i in range(5):
            print(f"Цифра {i+1}: Потери = {loss[i]:.4f}, Точность = {accuracy[i]:.4f}")


def main():
    """Основная функция для запуска обучения модели"""
    parser = argparse.ArgumentParser(description='Обучение модели для распознавания капч')
    
    # Параметры датасета
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Директория с датасетом')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Количество изображений для генерации (если датасет не существует)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Использовать оттенки серого вместо RGB')
    
    # Параметры обучения
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох обучения')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Директория для сохранения моделей')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Путь к существующей модели для продолжения обучения')
    
    # Параметры аугментации
    parser.add_argument('--augmentation', action='store_true',
                        help='Применять аугментации к обучающему датасету')
    parser.add_argument('--augmentation_factor', type=float, default=0.3,
                        help='Вероятность применения аугментаций (0-1)')
    
    # Дополнительные параметры
    parser.add_argument('--evaluate', action='store_true',
                        help='Оценить модель на тестовом датасете после обучения')
    
    args = parser.parse_args()
    
    # Создаем директорию для моделей, если её нет
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Включаем смешанную точность для ускорения обучения на GPU
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Включена смешанная точность для ускорения обучения")
    except:
        print("Смешанная точность не поддерживается на данном устройстве")
    
    # Обучаем модель
    train_model(args)


if __name__ == '__main__':
    main()