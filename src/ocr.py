#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PIL import Image
from model import CaptchaModel

class CaptchaOCR:
    """Простой OCR для распознавания 5-значных капч"""
    
    def __init__(self, model_path=None, grayscale=True):
        """Инициализация OCR
        
        Args:
            model_path (str): Путь к файлу модели. По умолчанию None - будет использована
                             модель по умолчанию из директории models/final_model.keras
            grayscale (bool): Использовать ли оттенки серого вместо RGB
        """
        if model_path is None:
            model_path = os.path.join('models', 'final_model.keras')
        
        # Создаем и загружаем модель
        self.model = CaptchaModel(grayscale=grayscale)
        self.model.load_model(model_path)
        print(f"Модель загружена из {model_path}")
    
    def recognize(self, image):
        """Распознает цифры на изображении капчи
        
        Args:
            image: Путь к файлу изображения или объект PIL.Image
            
        Returns:
            tuple: (text, confidences)
                text (str): Распознанный текст капчи (5 цифр)
                confidences (list): Список значений уверенности для каждой цифры
        """
        # Проверяем входные данные
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Файл не найден: {image}")
        elif not isinstance(image, Image.Image):
            raise TypeError("image должен быть путем к файлу или объектом PIL.Image")
        
        try:
            # Распознаем изображение
            text, confidences, _ = self.model.predict(image)
            return text, confidences
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при распознавании: {str(e)}")


def main():
    """Пример использования OCR"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR для распознавания капч')
    parser.add_argument('--image', type=str, required=True,
                      help='Путь к изображению капчи')
    parser.add_argument('--model', type=str, default=None,
                      help='Путь к файлу модели (по умолчанию models/final_model.keras)')
    parser.add_argument('--grayscale', action='store_true',
                      help='Использовать оттенки серого вместо RGB')
    
    args = parser.parse_args()
    
    # Создаем OCR
    ocr = CaptchaOCR(model_path=args.model, grayscale=args.grayscale)
    
    # Распознаем изображение
    try:
        text, confidences = ocr.recognize(args.image)
        print(f"\nРаспознанный текст: {text}")
        print("Уверенность для каждой цифры:")
        for i, conf in enumerate(confidences):
            print(f"Цифра {i+1}: {conf:.4f}")
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == '__main__':
    main()