#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import time


class CaptchaModel:
    """Модель нейросети для распознавания 5-значных капч"""
    
    def __init__(self, img_width=100, img_height=40, grayscale=True):
        """Инициализация модели
        
        Args:
            img_width (int): Ширина изображения
            img_height (int): Высота изображения
            grayscale (bool): Использовать ли оттенки серого (1 канал) или RGB (3 канала)
        """
        self.img_width = img_width
        self.img_height = img_height
        self.num_channels = 1 if grayscale else 3
        self.num_classes = 10  # Цифры от 0 до 9
        self.num_digits = 5    # 5 цифр в капче
        self.model = None
    
    def build_model(self):
        """Создает архитектуру модели
        
        Returns:
            tensorflow.keras.models.Model: Скомпилированная модель
        """
        # Входной слой
        input_layer = Input(shape=(self.img_height, self.img_width, self.num_channels))
        
        # Сверточные слои
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Полносвязные слои
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # 5 выходных слоев для каждой цифры
        outputs = []
        for i in range(self.num_digits):
            digit_output = Dense(self.num_classes, activation='softmax', name=f'digit_{i}')(x)
            outputs.append(digit_output)
        
        # Создаем модель
        model = Model(inputs=input_layer, outputs=outputs)
        
        # Компилируем модель
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['categorical_crossentropy'] * self.num_digits,
            metrics=['accuracy'] * self.num_digits
        )
        
        self.model = model
        return model
    
    def train(self, train_dataset, validation_dataset, epochs=50, batch_size=32, model_dir='models'):
        """Обучает модель
        
        Args:
            train_dataset (tf.data.Dataset): Обучающий датасет
            validation_dataset (tf.data.Dataset): Валидационный датасет
            epochs (int): Количество эпох обучения
            batch_size (int): Размер батча
            model_dir (str): Директория для сохранения моделей
            
        Returns:
            dict: История обучения
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Создаем модель, если она еще не создана
        if self.model is None:
            self.build_model()
        
        # Колбэки для обучения
        callbacks = [
            ModelCheckpoint(
                os.path.join(model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Обучаем модель
        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_dataset):
        """Оценивает модель на тестовом датасете
        
        Args:
            test_dataset (tf.data.Dataset): Тестовый датасет
            
        Returns:
            tuple: (loss, accuracy) - потери и точность для каждой цифры и общая точность
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите build_model()")
        
        # Оцениваем модель
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Первые self.num_digits значения - потери для каждой цифры
        # Следующие self.num_digits значения - точность для каждой цифры
        loss = results[:self.num_digits]
        accuracy = results[self.num_digits:]
        
        # Вычисляем общую точность (все 5 цифр правильно)
        total_correct = 0
        total_samples = 0
        
        for batch in test_dataset:
            images, true_labels = batch
            predictions = self.model.predict(images, verbose=0)
            
            # Преобразуем предсказания в индексы классов
            pred_indices = [np.argmax(predictions[i], axis=1) for i in range(self.num_digits)]
            true_indices = [np.argmax(true_labels[i], axis=1) for i in range(self.num_digits)]
            
            # Проверяем, все ли цифры угаданы правильно
            for i in range(len(images)):
                all_correct = True
                for j in range(self.num_digits):
                    if pred_indices[j][i] != true_indices[j][i]:
                        all_correct = False
                        break
                
                if all_correct:
                    total_correct += 1
                total_samples += 1
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return loss, accuracy, overall_accuracy
    
    def predict(self, image):
        """Распознает цифры на изображении капчи
        
        Args:
            image: Изображение капчи (путь к файлу или объект PIL.Image)
            
        Returns:
            str: Распознанная строка из 5 цифр
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите build_model()")
        
        # Импортируем препроцессор
        from preprocessing import CaptchaPreprocessor
        
        # Создаем препроцессор с теми же параметрами, что и модель
        preprocessor = CaptchaPreprocessor(
            img_width=self.img_width,
            img_height=self.img_height,
            grayscale=(self.num_channels == 1)
        )
        
        # Предобрабатываем изображение
        processed_image = preprocessor.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)  # Добавляем размерность батча
        
        # Замеряем время инференса
        start_time = time.time()
        
        # Получаем предсказания
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Вычисляем время инференса
        inference_time = time.time() - start_time
        
        # Преобразуем предсказания в цифры
        digits = [str(np.argmax(predictions[i][0])) for i in range(self.num_digits)]
        captcha_text = ''.join(digits)
        
        # Получаем уверенность для каждой цифры
        confidences = [np.max(predictions[i][0]) for i in range(self.num_digits)]
        
        return captcha_text, confidences, inference_time
    
    def save_model(self, filepath):
        """Сохраняет модель в файл
        
        Args:
            filepath (str): Путь для сохранения модели
        """
        if self.model is None:
            raise ValueError("Модель не создана. Сначала вызовите build_model()")
        
        self.model.save(filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath):
        """Загружает модель из файла
        
        Args:
            filepath (str): Путь к файлу модели
        """
        try:
            # Пробуем загрузить модель с обработкой кастомных объектов
            self.model = tf.keras.models.load_model(
                filepath,
                custom_objects=None,  # Здесь можно добавить кастомные слои если они есть
                compile=True,
                options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            )
            print(f"Модель успешно загружена из {filepath}")
        except Exception as e:
            print(f"Ошибка при загрузке модели: {str(e)}")
            try:
                # Пробуем альтернативный способ загрузки
                print("Пробуем альтернативный способ загрузки...")
                # Создаем новую модель с той же архитектурой
                self.build_model()
                # Загружаем только веса
                self.model.load_weights(filepath)
                print(f"Веса модели успешно загружены из {filepath}")
            except Exception as e2:
                raise ValueError(f"Не удалось загрузить модель: {str(e)}\nОшибка при альтернативной загрузке: {str(e2)}")


def main():
    """Тестирование модели"""
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Тестирование модели для распознавания капч')
    parser.add_argument('--train', action='store_true',
                        help='Обучить модель')
    parser.add_argument('--evaluate', action='store_true',
                        help='Оценить модель на тестовом датасете')
    parser.add_argument('--predict', type=str, default=None,
                        help='Путь к изображению для распознавания')
    parser.add_argument('--model_path', type=str, default='models/best_model.keras',
                        help='Путь к файлу модели')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Директория с датасетом')
    parser.add_argument('--grayscale', action='store_true',
                        help='Использовать оттенки серого вместо RGB')
    
    args = parser.parse_args()
    
    # Создаем модель
    model = CaptchaModel(grayscale=args.grayscale)
    
    if args.train:
        # Импортируем препроцессор
        from preprocessing import CaptchaPreprocessor
        
        # Создаем препроцессор
        preprocessor = CaptchaPreprocessor(grayscale=args.grayscale)
        
        # Загружаем данные
        train_dataset = preprocessor.load_data(
            os.path.join(args.dataset_dir, 'train'),
            batch_size=32,
            shuffle=True
        )
        
        # Применяем аугментации к обучающему датасету
        train_dataset = preprocessor.apply_augmentation(train_dataset)
        
        validation_dataset = preprocessor.load_data(
            os.path.join(args.dataset_dir, 'validation'),
            batch_size=32,
            shuffle=False
        )
        
        # Создаем и обучаем модель
        model.build_model()
        history = model.train(train_dataset, validation_dataset)
        
        # Визуализируем историю обучения
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Обучение')
        plt.plot(history.history['val_loss'], label='Валидация')
        plt.title('Функция потерь')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Вычисляем среднюю точность по всем выходам
        train_acc = np.mean([history.history[f'digit_{i}_accuracy'] for i in range(5)], axis=0)
        val_acc = np.mean([history.history[f'val_digit_{i}_accuracy'] for i in range(5)], axis=0)
        
        plt.plot(train_acc, label='Обучение')
        plt.plot(val_acc, label='Валидация')
        plt.title('Точность')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    elif args.evaluate:
        # Загружаем модель
        model.load_model(args.model_path)
        
        # Импортируем препроцессор
        from preprocessing import CaptchaPreprocessor
        
        # Создаем препроцессор
        preprocessor = CaptchaPreprocessor(grayscale=args.grayscale)
        
        # Загружаем тестовый датасет
        test_dataset = preprocessor.load_data(
            os.path.join(args.dataset_dir, 'test'),
            batch_size=32,
            shuffle=False
        )
        
        # Оцениваем модель
        loss, accuracy, overall_accuracy = model.evaluate(test_dataset)
        
        print("\nРезультаты оценки:")
        print(f"Общая точность (все 5 цифр): {overall_accuracy:.4f}")
        
        for i in range(5):
            print(f"Цифра {i+1}: Потери = {loss[i]:.4f}, Точность = {accuracy[i]:.4f}")
    
    elif args.predict is not None:
        # Загружаем модель
        model.load_model(args.model_path)
        
        # Распознаем изображение
        captcha_text, confidences, inference_time = model.predict(args.predict)
        
        print(f"\nРаспознанная капча: {captcha_text}")
        print(f"Время инференса: {inference_time:.4f} секунд")
        
        for i, (digit, conf) in enumerate(zip(captcha_text, confidences)):
            print(f"Цифра {i+1}: {digit} (уверенность: {conf:.4f})")
        
        # Отображаем изображение
        img = plt.imread(args.predict)
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.title(f"Распознано: {captcha_text} (время: {inference_time:.4f} с)")
        plt.axis('off')
        plt.show()
    
    else:
        print("Укажите действие: --train, --evaluate или --predict")


if __name__ == '__main__':
    import os
    main()