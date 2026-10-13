
import os
import warnings

# Hugging Face cache configuration
os.environ["HF_HOME"] = "D:/HuggingFace_Cache"
os.environ["HF_HUB_VERBOSITY"] = "error"

# Disable unnecessary warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\..*"
)

from transformers import pipeline, logging
import torch
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from scipy.io.wavfile import write

logging.set_verbosity_error()

# Загрузка фото
def load_image(file_path):
    path = Path(file_path)

    if not path.is_file():
        print("Ошибка: Файл не найден.")
        return None

    if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        print("Ошибка: Неподдерживаемый формат изображения.")
        return None

    try:
        with Image.open(path) as image:
            return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        print(f"Ошибка открытия изображения: {error}")
        return None




# Image analysis
def analyze_image(image, captioner):
    print("\nАнализ изображения...")

    try:
        result = captioner(image, max_new_tokens=50)

        if not result or not result[0].get("generated_text"):
            print("Ошибка: модель не создала описание.")
            return None

        description = result[0]["generated_text"].strip()

        if not description:
            print("Ошибка: получено пустое описание.")
            return None

        return description

    except Exception as error:
        print(f"Ошибка анализа изображения: {error}")
        return None



# Generate unique audio file path
def get_audio_path(file_name):
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)

    audio_path = output_folder / f"{file_name}.wav"

    counter = 1

    while audio_path.exists():
        audio_path = output_folder / f"{file_name}_{counter}.wav"
        counter += 1

    return audio_path


# Speech generation
def generate_speech(description, synthesizer, file_path, audio_name=None):
    print("\nГенерация голосового описания...")

    try:
        # 1. Сначала подготавливаем папку и проверяем имя (до запуска нейросети)
        output_folder = Path("output")
        output_folder.mkdir(exist_ok=True)

        if audio_name is None:
            audio_name = Path(file_path).stem

        audio_name = Path(audio_name).stem

        if not audio_name or audio_name in (".", ".."):
            print("Ошибка: некорректное название аудиофайла.")
            return None

        if any(char in audio_name for char in '<>:"/\\|?*'):
            print("Ошибка: название содержит недопустимые символы.")
            return None

        audio_path = get_audio_path(audio_name)

        # 2. теперь запускаем нейросеть один раз
        speech = synthesizer(description)
        
        # 3. Делаем проверки сгенерированного аудио
        if not speech or "audio" not in speech:
            print("Ошибка: модель не создала аудио.")
            return None

        audio = speech["audio"].squeeze()

        if audio.size == 0:
            print("Ошибка: получен пустой аудиофайл.")
            return None

        # 4. Сохраняем файл
        write(
            audio_path,
            speech["sampling_rate"],
            audio
        )

        return audio_path

    except Exception as error:
        print(f"Ошибка генерации аудио: {error}")
        return None



# Repeat speech generation
def repeat_speech(description, synthesizer, file_path):
    if description is None:
        print("\nОшибка: сначала необходимо проанализировать фотографию.")
        return

    print(f"\nПоследнее описание: {description}")

    audio_name = input("Введите название аудиофайла: ").strip()

    if not audio_name:
        print("Ошибка: название аудиофайла не может быть пустым.")
        return

    audio_path = generate_speech(
        description,
        synthesizer,
        file_path,
        audio_name
    )

    if audio_path is None:
        return

    print(f"\nАудиофайл успешно создан: {audio_path.resolve()}")



# Save text description
def save_description(file_path, description):
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)

    text_path = output_folder / "descriptions.txt"

    with open(text_path, "a", encoding="utf-8") as file:
        file.write(f"Image: {Path(file_path).name}\n")
        file.write(f"Description: {description}\n")
        file.write("-" * 40 + "\n")

    return text_path




# Model loading
def load_models():
    device = 0 if torch.cuda.is_available() else -1

    print("\nЗагрузка модели BLIP...")

    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=device
    )

    print("Загрузка модели MMS-TTS...")

    synthesizer = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device=device
    )

    return captioner, synthesizer




# Main program
def main():
    print("=== ИИ Ассистент ===")

    captioner, synthesizer = load_models()

    last_description = None
    last_file_path = None

    while True:
        print("\n========== МЕНЮ ==========")
        print("1. Анализ новой фотографии")
        print("2. Повторное озвучивание описания")
        print("3. Выход")
        print("==========================")

        choice = input("\nВыберите действие: ").strip()

        if choice == "1":
            file_path = input("\nВведите путь к изображению: ").strip().strip('"')

            image = load_image(file_path)

            if image is None:
                continue

            print("\nИзображение загружено успешно!")
            print(f"Имя изображения: {Path(file_path).name}")
            print(f"Размер изображения: {image.width}x{image.height}")

            description = analyze_image(image, captioner)

            if description is None:
                continue

            print(f"\nОписание изображения: {description}")

            audio_path = generate_speech(
                description,
                synthesizer,
                file_path
            )

            if audio_path is None:
                continue

            text_path = save_description(file_path, description)

            last_description = description
            last_file_path = file_path

            print("\n=== Результат ===")
            print(f"Фотография: {Path(file_path).name}")
            print(f"Описание: {description}")
            print("Модель анализа: Salesforce/blip-image-captioning-base")
            print("Модель озвучивания: facebook/mms-tts-eng")
            print(f"Аудиофайл: {audio_path.resolve()}")
            print(f"Текстовый файл: {text_path.resolve()}")

        elif choice == "2":
            repeat_speech(
                last_description,
                synthesizer,
                last_file_path
            )

        elif choice == "3":
            print("\nЗавершение программы.")
            break

        else:
            print("\nОшибка: выберите пункт 1, 2 или 3.")


if __name__ == "__main__":
    main()

