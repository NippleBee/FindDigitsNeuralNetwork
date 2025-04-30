#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import time


def run_pipeline(args):
    """Запускает полный пайплайн: генерация датасета, обучение, оценка и инференс
    
    Args:
        args: Аргументы командной строки
    """
    start_time = time.time()
    
    # Шаг 1: Генерация датасета
    if args.generate_dataset:
        print("\n=== Шаг 1: Генерация датасета ===")
        from data_generator import create_dataset
        
        create_dataset(
            num_samples=args.num_samples,
            output_dir=args.dataset_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
    
    # Шаг 2: Обучение модели
    if args.train_model:
        print("\n=== Шаг 2: Обучение модели ===")
        from train import train_model
        
        # Создаем аргументы для обучения
        train_args = argparse.Namespace(
            dataset_dir=args.dataset_dir,
            num_samples=args.num_samples,
            grayscale=args.grayscale,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=args.model_dir,
            load_model=args.load_model,
            augmentation=args.augmentation,
            augmentation_factor=args.augmentation_factor,
            evaluate=False  # Оценку будем делать отдельно
        )
        
        train_model(train_args)
    
    # Шаг 3: Оценка модели
    if args.evaluate_model:
        print("\n=== Шаг 3: Оценка модели ===")
        from inference import evaluate_model
        from model import CaptchaModel
        
        # Загружаем модель
        model_path = args.model_path or os.path.join(args.model_dir, 'best_model.h5')
        print(f"Загружаем модель из {model_path}...")
        
        model = CaptchaModel(grayscale=args.grayscale)
        model.load_model(model_path)
        
        # Оцениваем модель
        test_dir = os.path.join(args.dataset_dir, 'test')
        evaluate_model(model, test_dir, args.grayscale, args.output_dir)
    
    # Шаг 4: Инференс на тестовом изображении
    if args.test_image:
        print("\n=== Шаг 4: Инференс на тестовом изображении ===")
        from inference import predict_single_image
        from model import CaptchaModel
        
        # Загружаем модель, если еще не загружена
        if 'model' not in locals():
            model_path = args.model_path or os.path.join(args.model_dir, 'best_model.h5')
            print(f"Загружаем модель из {model_path}...")
            
            model = CaptchaModel(grayscale=args.grayscale)
            model.load_model(model_path)
        
        # Распознаем изображение
        predict_single_image(model, args.test_image, args.grayscale)
    
    # Выводим общее время выполнения
    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.2f} секунд")


def main():
    """Основная функция для запуска пайплайна"""
    parser = argparse.ArgumentParser(description='Пайплайн для распознавания 5-значных цифровых капч')
    
    # Общие параметры
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Директория с датасетом')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Директория для сохранения моделей')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--grayscale', action='store_true',
                        help='Использовать оттенки серого вместо RGB')
    
    # Параметры для генерации датасета
    parser.add_argument('--generate_dataset', action='store_true',
                        help='Генерировать датасет')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Количество изображений для генерации')
    
    # Параметры для обучения модели
    parser.add_argument('--train_model', action='store_true',
                        help='Обучать модель')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох обучения')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Путь к существующей модели для продолжения обучения')
    parser.add_argument('--augmentation', action='store_true',
                        help='Применять аугментации к обучающему датасету')
    parser.add_argument('--augmentation_factor', type=float, default=0.3,
                        help='Вероятность применения аугментаций (0-1)')
    
    # Параметры для оценки модели
    parser.add_argument('--evaluate_model', action='store_true',
                        help='Оценивать модель')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Путь к файлу модели (если не указан, используется best_model.h5 из model_dir)')
    
    # Параметры для инференса
    parser.add_argument('--test_image', type=str, default=None,
                        help='Путь к тестовому изображению для инференса')
    
    # Запуск всего пайплайна
    parser.add_argument('--run_all', action='store_true',
                        help='Запустить весь пайплайн (генерация, обучение, оценка)')
    
    args = parser.parse_args()
    
    # Если указан флаг run_all, включаем все этапы
    if args.run_all:
        args.generate_dataset = True
        args.train_model = True
        args.evaluate_model = True
    
    # Проверяем, что хотя бы один этап выбран
    if not any([args.generate_dataset, args.train_model, args.evaluate_model, args.test_image]):
        parser.print_help()
        print("\nОшибка: не выбран ни один этап. Укажите хотя бы один из флагов: --generate_dataset, --train_model, --evaluate_model, --test_image или --run_all")
        return
    
    # Создаем необходимые директории
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Запускаем пайплайн
    run_pipeline(args)


if __name__ == '__main__':
    main()