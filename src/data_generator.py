#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
import shutil


class CaptchaGenerator:
    """Генератор капч с 5 цифрами в прямоугольном стиле"""
    
    def __init__(self, width=220, height=90, bg_color_base=(117, 157, 163), # #759da3
                 text_color_base=(26, 36, 50), # #1a2432
                 digit_size=(35, 70), noise_level=0.0):
        """
        Примечание: Цвета подобраны на основе примеров изображений:
        - Фон: голубовато-серый
        - Текст: темно-синий
        """
        """Инициализация генератора капч
        
        Args:
            width (int): Ширина изображения
            height (int): Высота изображения
            bg_color_base (tuple): Базовый цвет фона (RGB)
            text_color_base (tuple): Базовый цвет текста (RGB)
            digit_size (tuple): Базовый размер цифры (ширина, высота)
            noise_level (float): Уровень шума (0-1)
        """
        self.width = width
        self.height = height
        self.bg_color_base = bg_color_base
        self.text_color_base = text_color_base
        self.base_digit_size = digit_size
        self.noise_level = noise_level
        self.line_thicknesses = [1, 1.25, 1.5]  # Варианты толщины линий
        
        # Определяем шаблоны для рисования цифр в прямоугольном стиле, как на примерах
        self.digit_patterns = {
            '0': [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ],
            '1': [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]
            ],
            '2': [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1]
            ],
            '3': [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ],
            '4': [
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1]
            ],
            '5': [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ],
            '6': [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ],
            '7': [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1]
            ],
            '8': [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ],
            '9': [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]
        }
    
    def _get_random_color(self, base_color, variation=20):
        """Генерирует случайный цвет с вариацией от базового
        
        Args:
            base_color (tuple): Базовый цвет (RGB)
            variation (int): Максимальное отклонение от базового цвета
            
        Returns:
            tuple: Случайный цвет (RGB)
        """
        r, g, b = base_color
        r = max(0, min(255, r + random.randint(-variation, variation)))
        g = max(0, min(255, g + random.randint(-variation, variation)))
        b = max(0, min(255, b + random.randint(-variation, variation)))
        return (r, g, b)
    
    def _add_noise(self, image):
        """Добавляет минимальный шум на изображение
        
        Args:
            image (PIL.Image): Исходное изображение
            
        Returns:
            PIL.Image: Изображение с шумом
        """
        # На примерах видно, что шума практически нет, поэтому делаем минимальный шум
        if self.noise_level <= 0:
            return image
            
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Добавляем очень мало точек
        num_dots = int(width * height * self.noise_level * 0.01)
        for _ in range(num_dots):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            r = 1  # Маленький размер точек
            color = self._get_random_color(self.text_color_base, variation=20)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
        
        return image
    
    def _apply_transformations(self, image):
        """Применяет минимальные трансформации к изображению
        
        Args:
            image (PIL.Image): Исходное изображение
            
        Returns:
            PIL.Image: Трансформированное изображение
        """
        # На примерах видно, что изображения не имеют искажений
        # Поэтому не применяем никаких трансформаций
        return image
    
    def _draw_digit(self, draw, digit, position, color):
        """Рисует цифру в прямоугольном стиле
        
        Args:
            draw (PIL.ImageDraw): Объект для рисования
            digit (str): Цифра для отрисовки (0-9)
            position (tuple): Позиция (x, y) для отрисовки
            color (tuple): Цвет цифры (RGB)
            
        Returns:
            int: Высота отрисованной цифры
        """
        pattern = self.digit_patterns[digit]
        base_width, base_height = self.base_digit_size
        
        # Случайная высота для цифры (±15% от базовой высоты)
        height_variation = random.uniform(-0.1, 0.1)
        digit_height = int(base_height * (1 + height_variation))
        
        # Используем базовую ширину
        digit_width = base_width
        
        x, y = position
        
        # Размер одного блока
        block_width = digit_width / 5
        block_height = digit_height / 7
        
        # Случайная толщина линий для этой цифры
        thickness = random.choice(self.line_thicknesses)
        
        # Рисуем цифру по шаблону
        for row_idx, row in enumerate(pattern):
            for col_idx, cell in enumerate(row):
                if cell == 1:
                    # Рассчитываем координаты блока с учетом толщины
                    block_x = x + col_idx * block_width
                    block_y = y + row_idx * block_height
                    
                    # Увеличиваем размер блока на толщину линии
                    adjusted_width = block_width * thickness
                    adjusted_height = block_height * thickness
                    
                    # Центрируем увеличенный блок
                    offset_x = (adjusted_width - block_width) / 2
                    offset_y = (adjusted_height - block_height) / 2
                    
                    # Рисуем прямоугольный блок с учетом толщины
                    draw.rectangle(
                        [
                            (block_x - offset_x, block_y - offset_y),
                            (block_x - offset_x + adjusted_width, block_y - offset_y + adjusted_height)
                        ],
                        fill=color
                    )
        
        return digit_height
    
    def generate_captcha(self):
        """Генерирует одно изображение капчи с 5 случайными цифрами в прямоугольном стиле
        
        Returns:
            tuple: (PIL.Image, str) - изображение капчи и строка с 5 цифрами
        """
        # Генерируем 5 случайных цифр
        digits = ''.join([str(random.randint(0, 9)) for _ in range(5)])
        
        # Создаем изображение с фоном случайного оттенка
        bg_color = self._get_random_color(self.bg_color_base)
        image = Image.new('RGB', (self.width, self.height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Рассчитываем позиции для цифр
        digit_width, base_height = self.base_digit_size
        spacing = (self.width - (5 * digit_width)) // 6  # Равномерное распределение
        
        # Рисуем каждую цифру
        for i, digit in enumerate(digits):
            # Случайное смещение по вертикали
            y_offset = random.randint(-2, 2)
            
            # Случайный цвет текста с вариацией
            text_color = self._get_random_color(self.text_color_base)
            
            # Позиция для текущей цифры
            x = spacing + i * (digit_width + spacing)
            
            # Вычисляем вертикальное положение
            y = (self.height - base_height) // 2 + y_offset
            
            # Рисуем цифру в правильной позиции
            self._draw_digit(draw, digit, (x, y), text_color)
        
        # Добавляем шум и трансформации
        image = self._add_noise(image)
        image = self._apply_transformations(image)
        
        return image, digits


def create_dataset(num_samples, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Создает датасет из указанного количества капч
    
    Args:
        num_samples (int): Количество изображений для генерации
        output_dir (str): Директория для сохранения датасета
        train_ratio (float): Доля обучающей выборки
        val_ratio (float): Доля валидационной выборки
        test_ratio (float): Доля тестовой выборки
    """
    # Проверяем, что соотношения корректны
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Соотношения выборок должны в сумме давать 1"
    
    # Создаем генератор капч
    generator = CaptchaGenerator()
    
    # Создаем директории для датасета
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    
    # Очищаем директории, если они существуют
    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    
    # Создаем файлы для меток
    train_labels = open(os.path.join(train_dir, 'labels.txt'), 'w')
    val_labels = open(os.path.join(val_dir, 'labels.txt'), 'w')
    test_labels = open(os.path.join(test_dir, 'labels.txt'), 'w')
    
    # Рассчитываем количество изображений для каждой выборки
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    print(f"Генерация датасета: {num_train} обучающих, {num_val} валидационных, {num_test} тестовых изображений")
    
    # Генерируем изображения
    for i in tqdm(range(num_samples), desc="Генерация капч"):
        image, digits = generator.generate_captcha()
        
        # Определяем, в какую выборку попадет изображение
        if i < num_train:
            save_dir = train_dir
            labels_file = train_labels
        elif i < num_train + num_val:
            save_dir = val_dir
            labels_file = val_labels
        else:
            save_dir = test_dir
            labels_file = test_labels
        
        # Сохраняем изображение
        image_path = os.path.join(save_dir, f"{i:06d}.png")
        image.save(image_path)
        
        # Записываем метку
        labels_file.write(f"{i:06d}.png {digits}\n")
    
    # Закрываем файлы с метками
    train_labels.close()
    val_labels.close()
    test_labels.close()
    
    print(f"Датасет успешно создан в директории {output_dir}")


def main():
    """Основная функция для запуска генерации датасета"""
    parser = argparse.ArgumentParser(description='Генератор датасета капч')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Количество изображений для генерации')
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Директория для сохранения датасета')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Доля обучающей выборки')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Доля валидационной выборки')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Доля тестовой выборки')
    
    args = parser.parse_args()
    
    # Создаем директорию для датасета, если её нет
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Генерируем датасет
    create_dataset(args.num_samples, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)


if __name__ == '__main__':
    main()