#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from src.data_generator import CaptchaGenerator
import matplotlib.pyplot as plt

def main():
    """Тестирование генератора капч"""
    parser = argparse.ArgumentParser(description='Тестирование генератора капч')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Количество изображений для генерации')
    parser.add_argument('--output_dir', type=str, default='test_output',
                        help='Директория для сохранения тестовых изображений')
    
    args = parser.parse_args()
    
    # Создаем директорию для тестовых изображений, если её нет
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Создаем генератор капч
    generator = CaptchaGenerator()
    
    # Генерируем и сохраняем изображения
    plt.figure(figsize=(15, 10))
    
    for i in range(args.num_samples):
        image, digits = generator.generate_captcha()
        
        # Сохраняем изображение
        image_path = os.path.join(args.output_dir, f"captcha_{i}_{digits}.png")
        image.save(image_path)
        
        # Отображаем изображение
        plt.subplot(1, args.num_samples, i + 1)
        plt.imshow(image)
        plt.title(digits)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'preview.png'))
    plt.show()
    
    print(f"Сгенерировано {args.num_samples} тестовых изображений в директории {args.output_dir}")

if __name__ == '__main__':
    main()