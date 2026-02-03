import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import io

# --- 1. SİSTEM AYARLARI ---
st.set_page_config(page_title="Mathrix Analysis Engine", layout="wide")

@st.cache_resource
def load_mathrix_engine():
    try:
        return load_model('lung_model.h5')
    except:
        return None

# --- 2. İLAÇ VE PROTOKOL VERİTABANI ---
TREATMENT_DATA = {
    'Adenocarcinoma': {
        'label': 'ADENOKARSİNOM TESPİT EDİLDİ',
        'drugs': ['Osimertinib', 'Gefitinib', 'Pemetrexed', 'Bevacizumab'],
        'info': 'Hedefe yönelik tedavi ve EGFR mutasyon analizi önerilir.',
        'color': 'red'
    },
    'Squamous Cell Carcinoma': {
        'label': 'SKUAMÖZ HÜCRELİ KANSER TESPİT EDİLDİ',
        'drugs': ['Cisplatin', 'Gemcitabine', 'Pembrolizumab', 'Docetaxel'],
        'info': 'Platin bazlı kemoterapi ve immünoterapi kombinasyonu uygundur.',
        'color': 'orange'
    },
    'Normal': {
        'label': 'SAĞLIKLI AKCİĞER DOKUSU',
        'drugs': ['İlaç Gerekli Değil'],
        'info': 'Hücresel boşluklar ve matris yapısı normal sınırlardadır.',
        'color': 'green'
    }
}

# --- 3. DERİN ANALİZ FONKSİYONU ---
def run_mathrix_analysis(img):
    # Boşlukları ve yoğunluğu sayalım
    img_gray = ImageOps.grayscale(img)
    arr = np.array(img_gray)
    density = np.mean(arr < 127) * 100 # Hücre yoğunluğu
    voids = np.mean(arr > 200) * 100    # Boşluk oranı
    return round(density, 2), round(voids, 2)

# --- 4. ARAYÜZ VE İŞLEME ---
st.title("🖥️ MATHRIX DERİN TEŞHİS VE İLAÇ SİSTEMİ")
st.write("---")

engine = load_mathrix_engine()
files = st.file_uploader("Analiz edilecek görselleri yükleyin", accept_multiple_files=True)

if files:
    if st.button("ANALİZİ BAŞLAT"):
        for file in files:
            img = Image.open(file).convert('RGB')
            
            # Arka planda gerçek matematiksel ölçüm
            density, voids = run_mathrix_analysis(img)
            
            # Model tahmini
            img_input = np.array(img.resize((224, 224))) / 255.0
            img_input = np.expand_dims(img_input, axis=0)
            
            if engine:
                preds = engine.predict(img_input)
                classes = ['Adenocarcinoma', 'Normal', 'Squamous Cell Carcinoma']
                res = classes[np.argmax(preds)]
                data = TREATMENT_DATA[res]
                
                # --- GÖRSEL ÇIKTI ---
                col1, col2 = st.columns(2)
                with col1:
                    # Resim üzerine bilgi kutusu ekleme
                    st.image(img, caption=f"Yoğunluk: %{density} | Boşluk: %{voids}", use_container_width=True)
                
                with col2:
                    if res == 'Normal':
                        st.success(f"### {data['label']}")
                    else:
                        st.error(f"### {data['label']}")
                        st.markdown("#### 💊 Önerilen İlaçlar:")
                        for drug in data['drugs']:
                            st.write(f"- {drug}")
                        st.info(f"💡 *Not:* {data['info']}")
                    
                    st.metric("Sistem Güven Skoru", f"%{np.max(preds)*100:.2f}")
                st.write("---")
