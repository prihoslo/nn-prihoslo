import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import random
import requests
from io import BytesIO
import torch.nn.functional as F
import torchvision.models as models

# Конфигурация страницы
st.set_page_config(
    page_title="Типы погодных явлений",
    page_icon="🌤️",
    layout="wide"
)

# Словарь классов
CLASS_NAMES = {
    0: 'dew',
    1: 'fogsmog',
    2: 'frost',
    3: 'glaze',
    4: 'hail',
    5: 'lightning',
    6: 'rain',
    7: 'rainbow',
    8: 'rime',
    9: 'sandstorm',
    10: 'snow'
}

# Названия классов на русском
CLASS_NAMES_RU = {
    0: 'роса',
    1: 'туман/смог',
    2: 'иней',
    3: 'гололёд',
    4: 'град',
    5: 'молния',
    6: 'дождь',
    7: 'радуга',
    8: 'изморозь',
    9: 'песчаная буря',
    10: 'снег'
}

# Цвета для прогресс-баров
CLASS_COLORS = {
    0: '#87CEEB',  # роса - небесно-голубой
    1: '#B0C4DE',  # туман - светло-стальной
    2: '#E0FFFF',  # иней - светлый циан
    3: '#F0F8FF',  # гололёд - алиса синий
    4: '#F5F5F5',  # град - белый дым
    5: '#FFD700',  # молния - золотой
    6: '#4682B4',  # дождь - стальной синий
    7: '#FF69B4',  # радуга - ярко-розовый
    8: '#F5F5DC',  # изморозь - бежевый
    9: '#D2B48C',  # песчаная буря - коричневый
    10: '#F0F8FF'  # снег - алиса синий
}

@st.cache_resource
def load_model():
    """Загрузка модели с правильными параметрами безопасности"""
    try:
        # Добавляем ShuffleNetV2 в список безопасных глобальных объектов
        torch.serialization.add_safe_globals([models.shufflenetv2.ShuffleNetV2])
        
        # Загружаем модель с weights_only=True (безопасный режим)
        model = torch.load('full_model.pth', map_location=torch.device('cpu'), weights_only=True)
        model.eval()
        return model
    except Exception as e:
        # Если не получается с weights_only=True, пробуем альтернативный метод
        try:
            model = torch.load('full_model.pth', map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            return model
        except:
            # Загружаем с weights_only=False как основной метод
            model = torch.load('full_model.pth', map_location=torch.device('cpu'), weights_only=False)
            model.eval()
            return model

def get_transform():
    """Трансформация для модели"""
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def predict_image(model, image, transform):
    """Предсказание для одного изображения"""
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=3)
    
    return {
        'top_indices': top_indices[0].tolist(),
        'top_probs': top_probs[0].tolist()
    }

def display_predictions(predictions, show_main_confidence=True):
    """Отображение предсказаний с красивым форматированием"""
    
    if show_main_confidence:
        main_conf = predictions['top_probs'][0] * 100
        st.markdown(f"### 🎯 Я уверен в предсказании на **{main_conf:.1f}%**")
        st.markdown("---")
    
    st.markdown("### 📊 Топ-3 предсказания:")
    
    for i in range(3):
        class_idx = predictions['top_indices'][i]
        class_name = CLASS_NAMES_RU[class_idx]
        confidence = predictions['top_probs'][i] * 100
        
        # Создаем колонки для лучшего отображения
        col1, col2, col3 = st.columns([2, 5, 1])
        
        with col1:
            if i == 0:
                st.markdown(f"**🥇 {i+1} место**")
            elif i == 1:
                st.markdown(f"**🥈 {i+1} место**")
            else:
                st.markdown(f"**🥉 {i+1} место**")
        
        with col2:
            # Прогресс-бар с кастомным цветом
            color = CLASS_COLORS.get(class_idx, '#3498db')
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="font-weight: bold; margin-bottom: 5px;">{class_name}</div>
                <div style="background-color: #f0f0f0; border-radius: 10px; height: 20px; width: 100%;">
                    <div style="background-color: {color}; border-radius: 10px; height: 20px; width: {confidence}%; 
                                display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; color: white; font-weight: bold;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"**{confidence:.1f}%**")

def load_random_images(valid_path, num_images=10):
    """Загрузка случайных изображений из папки valid"""
    all_images = []
    
    for class_name in os.listdir(valid_path):
        class_path = os.path.join(valid_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append({
                        'path': os.path.join(class_path, img_name),
                        'true_class': class_name
                    })
    
    # Выбираем случайные изображения
    selected = random.sample(all_images, min(num_images, len(all_images)))
    return selected

def load_image_from_url(url):
    """Загрузка изображения по URL"""
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        st.error(f"Ошибка загрузки изображения: {e}")
        return None

def main():
    st.title("🌤️ Типы погодных явлений")
    st.markdown("---")
    
    # Загрузка модели
    try:
        model = load_model()
        transform = get_transform()
        st.success("✅ Модель успешно загружена!")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        st.info("""
        **Возможные решения:**
        1. Убедитесь, что файл `full_model.pth` существует в текущей директории
        2. Проверьте версию PyTorch: `pip install torch==2.5.1` для совместимости
        """)
        st.stop()
    
    # Боковая панель для навигации
    st.sidebar.title("📋 Навигация")
    page = st.sidebar.radio(
        "Выберите режим:",
        ["📸 Случайные изображения", "🔗 Загрузить по ссылке", "📤 Загрузить своё изображение"]
    )
    
    # Информация о модели в сайдбаре
    with st.sidebar.expander("ℹ️ О модели", expanded=False):
        st.markdown("""
        **Архитектура:** ShuffleNetV2
        
        **Количество классов:** 11
        
        **Классы:**
        - 🌊 Роса (dew)
        - 🌫️ Туман/Смог (fogsmog)
        - ❄️ Иней (frost)
        - 🧊 Гололёд (glaze)
        - 🌨️ Град (hail)
        - ⚡ Молния (lightning)
        - 🌧️ Дождь (rain)
        - 🌈 Радуга (rainbow)
        - 🌬️ Изморозь (rime)
        - 🏜️ Песчаная буря (sandstorm)
        - 🌨️ Снег (snow)
        """)
    
    # Режим случайных изображений
    if page == "📸 Случайные изображения":
        st.header("📸 Случайные изображения из датасета")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🎲 Показать новые 10 изображений", use_container_width=True):
                st.rerun()
        
        valid_path = 'data/valid'
        
        if not os.path.exists(valid_path):
            st.error(f"❌ Папка {valid_path} не найдена!")
        else:
            images_data = load_random_images(valid_path, 10)
            
            if not images_data:
                st.warning("⚠️ Изображения не найдены в папке")
            else:
                # Отображаем изображения в сетке 2x5
                for row in range(2):
                    cols = st.columns(5)
                    
                    for col_idx in range(5):
                        idx = row * 5 + col_idx
                        if idx < len(images_data):
                            img_data = images_data[idx]
                            
                            with cols[col_idx]:
                                # Загружаем оригинальное изображение
                                original_img = Image.open(img_data['path']).convert('RGB')
                                
                                # Предсказание
                                pred = predict_image(model, original_img, transform)
                                
                                # Отображаем изображение
                                st.image(original_img, use_container_width=True)
                                
                                # Информация о предсказании
                                true_class_name = img_data['true_class']
                                pred_class_name = CLASS_NAMES_RU[pred['top_indices'][0]]
                                confidence = pred['top_probs'][0] * 100
                                
                                # Цвет текста в зависимости от правильности предсказания
                                is_correct = true_class_name == CLASS_NAMES[pred['top_indices'][0]]
                                color = "green" if is_correct else "red"
                                
                                st.markdown(f"""
                                <div style="font-size: 0.9em;">
                                    <b>Истинный класс:</b> {true_class_name}<br>
                                    <b style="color: {color};">Предсказание:</b> <span style="color: {color};">{pred_class_name}</span><br>
                                    <i>Уверенность: {confidence:.1f}%</i>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Показываем топ-3 в expander'е
                                with st.expander("📊 Топ-3 предсказания", expanded=False):
                                    for i in range(3):
                                        class_name = CLASS_NAMES_RU[pred['top_indices'][i]]
                                        conf = pred['top_probs'][i] * 100
                                        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                        st.markdown(f"{medal} **{class_name}**: {conf:.1f}%")
    
    # Режим загрузки по ссылке
    elif page == "🔗 Загрузить по ссылке":
        st.header("🔗 Загрузка изображения по ссылке")
        
        url = st.text_input("Введите URL изображения:", placeholder="https://example.com/image.jpg")
        
        if url:
            img = load_image_from_url(url)
            
            if img:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(img, caption="Загруженное изображение", use_container_width=True)
                
                with col2:
                    pred = predict_image(model, img, transform)
                    display_predictions(pred, show_main_confidence=True)
    
    # Режим загрузки своего изображения
    elif page == "📤 Загрузить своё изображение":
        st.header("📤 Загрузка своего изображения")
        
        uploaded_file = st.file_uploader(
            "Выберите изображение для анализа",
            type=['png', 'jpg', 'jpeg'],
            help="Поддерживаются форматы: PNG, JPG, JPEG"
        )
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, caption="Ваше изображение", use_container_width=True)
                
                # Информация о файле
                file_details = {
                    "Имя файла": uploaded_file.name,
                    "Тип файла": uploaded_file.type,
                    "Размер": f"{uploaded_file.size / 1024:.1f} KB"
                }
                st.json(file_details)
            
            with col2:
                pred = predict_image(model, img, transform)
                display_predictions(pred, show_main_confidence=True)

if __name__ == "__main__":
    main()