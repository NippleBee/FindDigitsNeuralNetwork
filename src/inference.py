#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Импортируем наши модули
from model import CaptchaModel
from preprocessing import CaptchaPreprocessor


def predict_single_image(model, image_path, grayscale=True):
    """Распознает цифры на одном изображении капчи
    
    Args:
        model (CaptchaModel): Модель для распознавания
        image_path (str): Путь к изображению
        grayscale (bool): Использовать ли оттенки серого
        
    Returns:
        tuple: (captcha_text, confidences, inference_time) - распознанный текст, 
               уверенность для каждой цифры и время инференса
    """
    # Распознаем изображение
    captcha_text, confidences, inference_time = model.predict(image_path)
    
    # Выводим результаты
    print(f"\nИзображение: {os.path.basename(image_path)}")
    print(f"Распознанная капча: {captcha_text}")
    print(f"Время инференса: {inference_time:.4f} секунд")
    
    for i, (digit, conf) in enumerate(zip(captcha_text, confidences)):
        print(f"Цифра {i+1}: {digit} (уверенность: {conf:.4f})")
    
    # Отображаем изображение
    img = plt.imread(image_path)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.title(f"Распознано: {captcha_text} (время: {inference_time:.4f} с)")
    plt.axis('off')
    
    return captcha_text, confidences, inference_time


def evaluate_model(model, test_dir, grayscale=True, output_dir='results'):
    """Оценивает модель на тестовом датасете
    
    Args:
        model (CaptchaModel): Модель для распознавания
        test_dir (str): Директория с тестовыми данными
        grayscale (bool): Использовать ли оттенки серого
        output_dir (str): Директория для сохранения результатов
        
    Returns:
        dict: Метрики качества модели
    """
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем метки из файла
    labels_file = os.path.join(test_dir, 'labels.txt')
    image_paths = []
    true_labels = []
    
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, label = parts
                image_path = os.path.join(test_dir, image_name)
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    true_labels.append(label)
    
    print(f"Загружено {len(image_paths)} тестовых изображений")
    
    # Распознаем все изображения
    pred_labels = []
    confidences = []
    inference_times = []
    
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Обработано {i}/{len(image_paths)} изображений")
        
        # Распознаем изображение
        captcha_text, conf, inf_time = model.predict(image_path)
        
        pred_labels.append(captcha_text)
        confidences.append(conf)
        inference_times.append(inf_time)
    
    # Вычисляем метрики
    correct_captchas = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    captcha_accuracy = correct_captchas / len(true_labels) if len(true_labels) > 0 else 0
    
    # Вычисляем точность для каждой позиции
    position_correct = [0] * 5
    position_total = len(true_labels)
    
    for true, pred in zip(true_labels, pred_labels):
        for i in range(5):
            if true[i] == pred[i]:
                position_correct[i] += 1
    
    position_accuracy = [correct / position_total for correct in position_correct]
    
    # Вычисляем confusion matrix для каждой позиции
    confusion_matrices = []
    for i in range(5):
        y_true = [int(label[i]) for label in true_labels]
        y_pred = [int(label[i]) for label in pred_labels]
        cm = confusion_matrix(y_true, y_pred, labels=range(10))
        confusion_matrices.append(cm)
    
    # Вычисляем среднее время инференса
    avg_inference_time = np.mean(inference_times)
    
    # Выводим результаты
    print("\nРезультаты оценки:")
    print(f"Общая точность (все 5 цифр): {captcha_accuracy:.4f}")
    
    for i in range(5):
        print(f"Точность для позиции {i+1}: {position_accuracy[i]:.4f}")
    
    print(f"Среднее время инференса: {avg_inference_time:.4f} секунд")
    
    # Визуализируем confusion matrices
    plt.figure(figsize=(20, 15))
    
    for i in range(5):
        plt.subplot(2, 3, i+1)
        sns.heatmap(confusion_matrices[i], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix для позиции {i+1}')
        plt.xlabel('Предсказанная цифра')
        plt.ylabel('Истинная цифра')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    
    # Визуализируем распределение времени инференса
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=20)
    plt.axvline(x=avg_inference_time, color='r', linestyle='--', label=f'Среднее: {avg_inference_time:.4f} с')
    plt.axvline(x=1.0, color='g', linestyle='--', label='Требование: 1.0 с')
    plt.title('Распределение времени инференса')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Количество изображений')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'inference_time.png'))
    
    # Сохраняем примеры ошибок
    error_dir = os.path.join(output_dir, 'errors')
    os.makedirs(error_dir, exist_ok=True)
    
    # Сохраняем до 50 примеров ошибок
    error_count = 0
    for i, (true, pred, image_path) in enumerate(zip(true_labels, pred_labels, image_paths)):
        if true != pred and error_count < 50:
            # Копируем изображение в директорию с ошибками
            img = Image.open(image_path)
            error_path = os.path.join(error_dir, f"error_{error_count}_true_{true}_pred_{pred}.png")
            img.save(error_path)
            error_count += 1
    
    print(f"Сохранено {error_count} примеров ошибок в {error_dir}")
    
    # Возвращаем метрики
    metrics = {
        'captcha_accuracy': captcha_accuracy,
        'position_accuracy': position_accuracy,
        'avg_inference_time': avg_inference_time,
        'confusion_matrices': confusion_matrices
    }
    
    return metrics


def main():
    """Основная функция для запуска инференса и оценки модели"""
    parser = argparse.ArgumentParser(description='Инференс и оценка модели для распознавания капч')
    
    # Режимы работы
    parser.add_argument('--mode', type=str, choices=['predict', 'evaluate'], default='predict',
                        help='Режим работы: predict - распознавание одного изображения, evaluate - оценка на тестовом датасете')
    
    # Параметры модели
    parser.add_argument('--model_path', type=str, default='models/best_model.keras',
                        help='Путь к файлу модели')
    parser.add_argument('--grayscale', action='store_true',
                        help='Использовать оттенки серого вместо RGB')
    
    # Параметры для режима predict
    parser.add_argument('--image_path', type=str, default=None,
                        help='Путь к изображению для распознавания (для режима predict)')
    
    # Параметры для режима evaluate
    parser.add_argument('--test_dir', type=str, default='dataset/test',
                        help='Директория с тестовыми данными (для режима evaluate)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Директория для сохранения результатов (для режима evaluate)')
    
    args = parser.parse_args()
    
    # Загружаем модель
    print(f"Загружаем модель из {args.model_path}...")
    model = CaptchaModel(grayscale=args.grayscale)
    model.load_model(args.model_path)
    
    if args.mode == 'predict':
        if args.image_path is None:
            print("Ошибка: не указан путь к изображению для режима predict")
            return
        
        # Распознаем одно изображение
        predict_single_image(model, args.image_path, args.grayscale)
        plt.show()
    
    elif args.mode == 'evaluate':
        # Оцениваем модель на тестовом датасете
        evaluate_model(model, args.test_dir, args.grayscale, args.output_dir)
        plt.show()


if __name__ == '__main__':
    main()