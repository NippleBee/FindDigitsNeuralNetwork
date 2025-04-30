#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


class CaptchaPreprocessor:
    """Класс для предобработки изображений капч"""
    
    def __init__(self, img_width=100, img_height=40, grayscale=True):
        """Инициализация препроцессора
        
        Args:
            img_width (int): Ширина изображения
            img_height (int): Высота изображения
            grayscale (bool): Преобразовывать ли в оттенки серого
        """
        self.img_width = img_width
        self.img_height = img_height
        self.grayscale = grayscale
        self.num_classes = 10  # Цифры от 0 до 9
        self.num_digits = 5    # 5 цифр в капче
    
    def preprocess_image(self, image_path):
        """Предобработка одного изображения
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            numpy.ndarray: Предобработанное изображение
        """
        # Загружаем изображение
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            # Если передан объект PIL.Image
            img = np.array(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # PIL использует RGB, OpenCV - BGR
        
        # Изменяем размер
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Преобразуем в оттенки серого, если нужно
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)  # Добавляем канал
        
        # Нормализуем значения пикселей в диапазон [0, 1]
        img = img.astype('float32') / 255.0
        
        return img
    
    def load_data(self, data_dir, batch_size=32, shuffle=True):
        """Загружает данные из директории и создает tf.data.Dataset
        
        Args:
            data_dir (str): Путь к директории с данными
            batch_size (int): Размер батча
            shuffle (bool): Перемешивать ли данные
            
        Returns:
            tf.data.Dataset: Датасет для обучения/валидации/тестирования
        """
        # Загружаем метки из файла
        labels_file = os.path.join(data_dir, 'labels.txt')
        image_paths = []
        labels = []
        
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, label = parts
                    image_path = os.path.join(data_dir, image_name)
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                        labels.append(label)
        
        # Создаем генератор данных
        def generator():
            for image_path, label in zip(image_paths, labels):
                # Предобрабатываем изображение
                img = self.preprocess_image(image_path)
                
                # Преобразуем метку в one-hot encoding для каждой цифры
                label_digits = [int(digit) for digit in label]
                label_one_hot = [to_categorical(digit, num_classes=self.num_classes) for digit in label_digits]
                
                yield img, tuple(label_one_hot)
        
        # Определяем типы и формы данных
        if self.grayscale:
            img_shape = (self.img_height, self.img_width, 1)
        else:
            img_shape = (self.img_height, self.img_width, 3)
        
        output_types = (tf.float32, tuple([tf.float32] * self.num_digits))
        output_shapes = (img_shape, tuple([(self.num_classes,)] * self.num_digits))
        
        # Создаем датасет
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        
        # Перемешиваем и формируем батчи
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        return dataset
    
    def apply_augmentation(self, dataset, augmentation_factor=0.3):
        """Применяет аугментации к датасету
        
        Args:
            dataset (tf.data.Dataset): Исходный датасет
            augmentation_factor (float): Вероятность применения аугментаций
            
        Returns:
            tf.data.Dataset: Датасет с аугментациями
        """
        def augment(image, labels):
            # Случайное изменение яркости
            if tf.random.uniform(()) < augmentation_factor:
                image = tf.image.random_brightness(image, max_delta=0.2)
            
            # Случайное изменение контраста
            if tf.random.uniform(()) < augmentation_factor:
                image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            
            # Добавление случайного шума
            if tf.random.uniform(()) < augmentation_factor:
                noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
                image = tf.clip_by_value(image + noise, 0.0, 1.0)
            
            return image, labels
        
        return dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def main():
    """Тестирование препроцессора"""
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование препроцессора капч')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                        help='Директория с данными для тестирования')
    parser.add_argument('--grayscale', action='store_true',
                        help='Преобразовывать ли в оттенки серого')
    
    args = parser.parse_args()
    
    # Создаем препроцессор
    preprocessor = CaptchaPreprocessor(grayscale=args.grayscale)
    
    # Загружаем данные
    dataset = preprocessor.load_data(args.data_dir, batch_size=5)
    
    # Применяем аугментации
    augmented_dataset = preprocessor.apply_augmentation(dataset)
    
    # Визуализируем несколько примеров
    plt.figure(figsize=(15, 10))
    
    for i, (images, labels) in enumerate(dataset.take(1)):
        for j in range(min(5, len(images))):
            plt.subplot(2, 5, j + 1)
            
            # Преобразуем изображение для отображения
            img = images[j].numpy()
            if args.grayscale:
                img = np.squeeze(img, axis=-1)  # Убираем канал для grayscale
            
            plt.imshow(img, cmap='gray' if args.grayscale else None)
            
            # Получаем метки
            digit_labels = [np.argmax(labels[k][j].numpy()) for k in range(5)]
            plt.title(''.join(map(str, digit_labels)))
            plt.axis('off')
    
    # Визуализируем аугментированные примеры
    for i, (images, labels) in enumerate(augmented_dataset.take(1)):
        for j in range(min(5, len(images))):
            plt.subplot(2, 5, j + 6)
            
            # Преобразуем изображение для отображения
            img = images[j].numpy()
            if args.grayscale:
                img = np.squeeze(img, axis=-1)  # Убираем канал для grayscale
            
            plt.imshow(img, cmap='gray' if args.grayscale else None)
            
            # Получаем метки
            digit_labels = [np.argmax(labels[k][j].numpy()) for k in range(5)]
            plt.title('Augmented: ' + ''.join(map(str, digit_labels)))
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_examples.png')
    plt.show()


if __name__ == '__main__':
    main()