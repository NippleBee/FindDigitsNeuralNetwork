#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil


def setup_project_structure(args):
    """Создает структуру проекта для нейросети по распознаванию капч
    
    Args:
        args: Аргументы командной строки
    """
    # Основные директории проекта
    directories = [
        args.dataset_dir,
        os.path.join(args.dataset_dir, 'train'),
        os.path.join(args.dataset_dir, 'validation'),
        os.path.join(args.dataset_dir, 'test'),
        'src',
        args.model_dir,
        'results',
        'fonts'
    ]
    
    # Создаем директории
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Создана директория: {directory}")
    
    # Проверяем наличие исходных файлов
    src_files = [
        'data_generator.py',
        'preprocessing.py',
        'model.py',
        'train.py',
        'inference.py',
        'run_pipeline.py'
    ]
    
    missing_files = []
    for file in src_files:
        if not os.path.exists(os.path.join('src', file)):
            missing_files.append(file)
    
    if missing_files:
        print("\nВНИМАНИЕ: Следующие файлы отсутствуют в директории src:")
        for file in missing_files:
            print(f"  - {file}")
        print("Убедитесь, что все необходимые файлы скопированы в директорию src.")
    
    print("\nСтруктура проекта успешно создана!")
    print("\nДля генерации датасета выполните:")
    print(f"python src/data_generator.py --num_samples {args.num_samples} --output_dir {args.dataset_dir}")
    print("\nДля запуска полного пайплайна выполните:")
    print(f"python src/run_pipeline.py --run_all --num_samples {args.num_samples} --epochs {args.epochs} --augmentation")


def main():
    """Основная функция для настройки структуры проекта"""
    parser = argparse.ArgumentParser(description='Настройка структуры проекта для нейросети по распознаванию капч')
    
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Директория для хранения датасета')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Директория для сохранения моделей')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Рекомендуемое количество изображений для генерации')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Рекомендуемое количество эпох обучения')
    parser.add_argument('--clean', action='store_true',
                        help='Очистить существующие директории перед созданием новых')
    
    args = parser.parse_args()
    
    # Очищаем директории, если указан флаг --clean
    if args.clean:
        directories_to_clean = [args.dataset_dir, args.model_dir, 'results']
        for directory in directories_to_clean:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Директория {directory} очищена")
    
    # Создаем структуру проекта
    setup_project_structure(args)


if __name__ == '__main__':
    main()