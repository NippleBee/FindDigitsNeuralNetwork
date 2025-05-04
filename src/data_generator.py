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
    
    def __init__(self, full_width=1050, full_height=150, captcha_width=515, captcha_height=150,
                 bg_color_base=(117, 157, 163), # #759da3
                 text_color_base=(26, 36, 50), # #1a2432
                 digit_size=(85, 115), noise_level=0.2, distraction_level=0.5, downscale_factor=1.0):
        """
        Примечание: Цвета подобраны на основе примеров изображений:
        - Фон: голубовато-серый
        - Текст: темно-синий
        """
        """Инициализация генератора капч
        
        Args:
            full_width (int): Полная ширина изображения
            full_height (int): Полная высота изображения
            captcha_width (int): Ширина области с капчей
            captcha_height (int): Высота области с капчей
            bg_color_base (tuple): Базовый цвет фона (RGB)
            text_color_base (tuple): Базовый цвет текста (RGB)
            digit_size (tuple): Базовый размер цифры (ширина, высота)
            noise_level (float): Уровень шума (0-1)
            distraction_level (float): Уровень помех вокруг капчи (0-1)
            downscale_factor (float): Коэффициент уменьшения размера (1.0 - без изменений)
        """
        self.full_width = full_width
        self.full_height = full_height
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        self.bg_color_base = bg_color_base
        self.text_color_base = text_color_base
        self.base_digit_size = digit_size
        self.noise_level = noise_level
        self.distraction_level = distraction_level
        self.downscale_factor = downscale_factor
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
    
    def _add_distractions(self, image, captcha_x=None, captcha_width=None):
        """Добавляет помехи вокруг основной капчи
        
        Args:
            image (PIL.Image): Исходное изображение
            captcha_x (int, optional): X-координата капчи. По умолчанию None.
            captcha_width (int, optional): Ширина капчи с обводкой. По умолчанию None.
            
        Returns:
            PIL.Image: Изображение с помехами
        """
        if self.distraction_level <= 0:
            return image
            
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Определяем области для помех (все, кроме области капчи)
        if captcha_x is None:
            captcha_x = (width - self.captcha_width) // 2
        if captcha_width is None:
            captcha_width = self.captcha_width
            
        # Рассчитываем Y-координату капчи (для вертикального позиционирования)
        captcha_height_with_border = self.captcha_height + 20  # 20 - это удвоенная толщина обводки (10*2)
        captcha_y = max(0, height - captcha_height_with_border)
        
        # Увеличиваем количество элементов помех для более значимого эффекта
        num_elements = int((width * height - captcha_width * captcha_height_with_border) * self.distraction_level * 0.003)
        
        # Добавляем различные типы помех
        for _ in range(num_elements):
            # Выбираем случайную область вне капчи
            valid_position = False
            while not valid_position:
                if random.random() < 0.5:  # Левая сторона по горизонтали
                    x = random.randint(0, captcha_x - 1) if captcha_x > 0 else 0
                    y = random.randint(0, height - 1)
                    valid_position = True
                else:  # Правая сторона по горизонтали
                    x = random.randint(captcha_x + captcha_width, width - 1) if captcha_x + captcha_width < width else width - 1
                    y = random.randint(0, height - 1)
                    valid_position = True
                
                # Проверяем, не попадает ли точка в область капчи
                if captcha_x <= x < captcha_x + captcha_width and captcha_y <= y < captcha_y + captcha_height_with_border:
                    valid_position = False
            
            # Выбираем тип помехи с разными вероятностями
            distraction_type = random.choices(
                ['line', 'rectangle', 'circle', 'text', 'filled_rectangle', 'filled_circle', 'pattern'],
                weights=[15, 20, 15, 15, 15, 10, 10],
                k=1
            )[0]
            
            # Случайный цвет для помехи с большей вариацией
            color = self._get_random_color(self.text_color_base, variation=80)
            # Иногда используем яркие цвета для большей заметности
            if random.random() < 0.3:
                bright_colors = [
                    (255, 0, 0),    # Красный
                    (0, 255, 0),    # Зеленый
                    (0, 0, 255),    # Синий
                    (255, 255, 0),  # Желтый
                    (255, 0, 255),  # Пурпурный
                    (0, 255, 255),  # Голубой
                    (255, 165, 0),  # Оранжевый
                    (128, 0, 128)   # Фиолетовый
                ]
                color = random.choice(bright_colors)
            
            if distraction_type == 'line':
                # Рисуем линию с большей длиной и толщиной
                length = random.randint(20, 100)
                angle = random.uniform(0, 2 * 3.14159)  # Случайный угол в радианах
                end_x = x + int(length * np.cos(angle))
                end_y = y + int(length * np.sin(angle))
                draw.line([(x, y), (end_x, end_y)], fill=color, width=random.randint(2, 5))
                
            elif distraction_type == 'rectangle':
                # Рисуем прямоугольник
                width_rect = random.randint(10, 50)
                height_rect = random.randint(10, 50)
                draw.rectangle([(x, y), (x + width_rect, y + height_rect)], 
                               outline=color, width=random.randint(2, 4))
                
            elif distraction_type == 'circle':
                # Рисуем круг
                radius = random.randint(10, 30)
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], 
                             outline=color, width=random.randint(2, 4))
                
            elif distraction_type == 'text':
                # Рисуем случайный текст (цифры или буквы)
                chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                # Иногда рисуем несколько символов
                text_length = random.choices([1, 2, 3], weights=[60, 30, 10], k=1)[0]
                text = ''.join(random.choice(chars) for _ in range(text_length))
                # Используем более крупный шрифт
                font_size = random.randint(15, 30)
                try:
                    # Пытаемся использовать TrueType шрифт, если доступен
                    font = ImageFont.truetype("arial.ttf", font_size)
                    draw.text((x, y), text, fill=color, font=font)
                except IOError:
                    # Если шрифт недоступен, используем стандартный
                    draw.text((x, y), text, fill=color)
                    
            elif distraction_type == 'filled_rectangle':
                # Рисуем закрашенный прямоугольник
                width_rect = random.randint(10, 40)
                height_rect = random.randint(10, 40)
                # Используем полупрозрачный цвет для заливки
                fill_color = color + (random.randint(50, 150),)  # Добавляем альфа-канал
                try:
                    # Создаем временное изображение с альфа-каналом
                    temp_img = Image.new('RGBA', image.size, (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    temp_draw.rectangle([(x, y), (x + width_rect, y + height_rect)], fill=fill_color)
                    # Накладываем на основное изображение
                    image = Image.alpha_composite(image.convert('RGBA'), temp_img).convert('RGB')
                    draw = ImageDraw.Draw(image)  # Обновляем объект для рисования
                except Exception:
                    # Если не удалось использовать альфа-канал, рисуем обычный прямоугольник
                    draw.rectangle([(x, y), (x + width_rect, y + height_rect)], fill=color)
                
            elif distraction_type == 'filled_circle':
                # Рисуем закрашенный круг
                radius = random.randint(10, 25)
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=color)
                
            elif distraction_type == 'pattern':
                # Рисуем узор из линий или точек
                pattern_size = random.randint(30, 60)
                pattern_type = random.choice(['grid', 'dots', 'diagonal'])
                
                if pattern_type == 'grid':
                    # Рисуем сетку
                    spacing = random.randint(5, 10)
                    for i in range(0, pattern_size, spacing):
                        draw.line([(x, y + i), (x + pattern_size, y + i)], fill=color, width=1)
                        draw.line([(x + i, y), (x + i, y + pattern_size)], fill=color, width=1)
                        
                elif pattern_type == 'dots':
                    # Рисуем точки
                    spacing = random.randint(5, 10)
                    dot_radius = random.randint(1, 3)
                    for i in range(0, pattern_size, spacing):
                        for j in range(0, pattern_size, spacing):
                            if x + i < width and y + j < height:
                                draw.ellipse([(x + i - dot_radius, y + j - dot_radius), 
                                             (x + i + dot_radius, y + j + dot_radius)], fill=color)
                                
                elif pattern_type == 'diagonal':
                    # Рисуем диагональные линии
                    spacing = random.randint(5, 10)
                    for i in range(0, pattern_size * 2, spacing):
                        draw.line([(x, y + i), (x + i, y)], fill=color, width=1)
        
        return image
    
    def generate_captcha(self):
        """Генерирует одно изображение капчи с 5 случайными цифрами в прямоугольном стиле
        
        Returns:
            tuple: (PIL.Image, str) - изображение капчи и строка с 5 цифрами
        """
        # Генерируем 5 случайных цифр
        digits = ''.join([str(random.randint(0, 9)) for _ in range(5)])
        
        # Создаем полное изображение с фоном случайного оттенка
        bg_color = self._get_random_color(self.bg_color_base)
        full_image = Image.new('RGB', (self.full_width, self.full_height), bg_color)
        
        # Создаем изображение для капчи
        captcha_bg_color = self._get_random_color(self.bg_color_base)
        captcha_image = Image.new('RGB', (self.captcha_width, self.captcha_height), captcha_bg_color)
        captcha_draw = ImageDraw.Draw(captcha_image)
        
        # Рассчитываем позиции для цифр
        digit_width, base_height = self.base_digit_size
        spacing = (self.captcha_width - (5 * digit_width)) // 6  # Равномерное распределение
        
        # Рисуем каждую цифру
        for i, digit in enumerate(digits):
            # Случайное смещение по вертикали
            y_offset = random.randint(-2, 2)
            
            # Случайный цвет текста с вариацией
            text_color = self._get_random_color(self.text_color_base)
            
            # Позиция для текущей цифры
            x = spacing + i * (digit_width + spacing)
            
            # Вычисляем вертикальное положение
            y = (self.captcha_height - base_height) // 2 + y_offset
            
            # Рисуем цифру в правильной позиции
            self._draw_digit(captcha_draw, digit, (x, y), text_color)
        
        # Минимизируем или полностью убираем шум внутри окна с капчей
        # Если нужен минимальный шум, используем очень низкий уровень шума
        original_noise_level = self.noise_level
        self.noise_level = self.noise_level * 0.1  # Уменьшаем шум в 10 раз
        captcha_image = self._add_noise(captcha_image)
        self.noise_level = original_noise_level  # Восстанавливаем исходный уровень шума
        
        captcha_image = self._apply_transformations(captcha_image)
        
        # Создаем изображение с обводкой для капчи
        border_width = 10  # Толщина обводки в пикселях
        captcha_with_border_width = self.captcha_width + 2 * border_width
        captcha_with_border_height = self.captcha_height + 2 * border_width
        
        # Создаем изображение с обводкой
        border_color = self._get_random_color(self.text_color_base, variation=30)
        captcha_with_border = Image.new('RGB', (captcha_with_border_width, captcha_with_border_height), border_color)
        
        # Вставляем капчу внутрь обводки
        captcha_with_border.paste(captcha_image, (border_width, border_width))
        
        # Случайное позиционирование по горизонтали (по X)
        # Вычисляем максимально возможное смещение, чтобы капча не выходила за границы
        max_x_offset = self.full_width - captcha_with_border_width
        if max_x_offset > 0:
            captcha_x = random.randint(0, max_x_offset)
        else:
            captcha_x = 0
        
        # Добавляем помехи вокруг капчи (до вставки капчи, чтобы она была на переднем плане)
        full_image = self._add_distractions(full_image, captcha_x, captcha_with_border_width)
        
        # Вставляем капчу с обводкой в полное изображение (поверх помех)
        # Рассчитываем Y-координату так, чтобы нижняя граница помещалась в изображение
        captcha_y = max(0, self.full_height - captcha_with_border_height)
        full_image.paste(captcha_with_border, (captcha_x, captcha_y))
        
        # Применяем даунскейлинг, если нужно
        if self.downscale_factor < 1.0:
            new_width = int(self.full_width * self.downscale_factor)
            new_height = int(self.full_height * self.downscale_factor)
            full_image = full_image.resize((new_width, new_height), Image.LANCZOS)
        
        return full_image, digits


def create_dataset(num_samples, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                  full_width=1050, full_height=150, captcha_width=515, captcha_height=150,
                  noise_level=0.2, distraction_level=0.5, downscale_factor=1.0):
    """Создает датасет из указанного количества капч
    
    Args:
        num_samples (int): Количество изображений для генерации
        output_dir (str): Директория для сохранения датасета
        train_ratio (float): Доля обучающей выборки
        val_ratio (float): Доля валидационной выборки
        test_ratio (float): Доля тестовой выборки
        full_width (int): Полная ширина изображения
        full_height (int): Полная высота изображения
        captcha_width (int): Ширина области с капчей
        captcha_height (int): Высота области с капчей
        noise_level (float): Уровень шума (0-1)
        distraction_level (float): Уровень помех вокруг капчи (0-1)
        downscale_factor (float): Коэффициент уменьшения размера (1.0 - без изменений)
    """
    # Проверяем, что соотношения корректны
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Соотношения выборок должны в сумме давать 1"
    
    # Создаем генератор капч с новыми параметрами
    generator = CaptchaGenerator(
        full_width=full_width,
        full_height=full_height,
        captcha_width=captcha_width,
        captcha_height=captcha_height,
        noise_level=noise_level,
        distraction_level=distraction_level,
        downscale_factor=downscale_factor
    )
    
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
    parser.add_argument('--full_width', type=int, default=1050,
                        help='Полная ширина изображения')
    parser.add_argument('--full_height', type=int, default=150,
                        help='Полная высота изображения')
    parser.add_argument('--captcha_width', type=int, default=515,
                        help='Ширина области с капчей')
    parser.add_argument('--captcha_height', type=int, default=150,
                        help='Высота области с капчей')
    parser.add_argument('--noise_level', type=float, default=0.2,
                        help='Уровень шума (0-1)')
    parser.add_argument('--distraction_level', type=float, default=0.5,
                        help='Уровень помех вокруг капчи (0-1)')
    parser.add_argument('--downscale_factor', type=float, default=1.0,
                        help='Коэффициент уменьшения размера (1.0 - без изменений)')
    
    args = parser.parse_args()
    
    # Создаем директорию для датасета, если её нет
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Генерируем датасет с новыми параметрами
    create_dataset(
        args.num_samples, 
        args.output_dir, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio,
        args.full_width,
        args.full_height,
        args.captcha_width,
        args.captcha_height,
        args.noise_level,
        args.distraction_level,
        args.downscale_factor
    )


if __name__ == '__main__':
    main()