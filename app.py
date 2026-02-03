import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Mathrix Vision Core", layout="wide")

@st.cache_resource
def load_mathrix_engine():
    try:
        return load_model('lung_model.h5')
    except:
        return None

# --- GENİŞLETİLMİŞ İLAÇ VE TEDAVİ VERİTABANI ---
DRUG_DATABASE = {
    'Adenocarcinoma': {
        'label': 'ADENOKARSİNOM TESPİT EDİLDİ',
        'drugs': ['Gefitinib (Iressa)', 'Erlotinib (Tarceva)', 'Afatinib (Gilotrif)', 'Osimertinib (Tagrisso)', 'Pemetrexed (Alimta)'],
        'therapy': 'Hedefe Yönelik TKİ Terapisi + İmmünoterapi',
        'color': (255, 0, 0) # Kırmızı
    },
    'Squamous Cell Carcinoma': {
        'label': 'SKUAMÖZ HÜCRELİ KANSER TESPİT EDİLDİ',
        'drugs': ['Cisplatin', 'Gemcitabine (Gemzar)', 'Paclitaxel (Taxol)', 'Pembrolizumab (Keytruda)', 'Necitumumab'],
        'therapy': 'Platin Bazlı Kombinasyon + Monoklonal Antikor',
        'color': (255, 165, 0) # Turuncu
    },
    'Normal': {
        'label': 'SAĞLIKLI DOKU ANALİZİ',
        'drugs': ['İlaç Gerekli Değil'],
        'therapy': 'Yıllık Rutin Radyolojik Takip',
        'color': (0, 255, 0) # Yeşil
    }
}

# --- GÖRÜNTÜ ÜZERİNE ANALİZ BALONU EKLEME ---
def draw_analysis_overlay(img, label, color):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    # Resmin üzerine teknolojik bir çerçeve ve "Analiz Bulutu" çizelim
    draw.rectangle([10, 10, width-10, height-10], outline=color, width=5)
    draw.text((20, 20), f"SCANNING: {label}", fill=color)
    return img

# --- ANA EKRAN ---
st.title("🖥️ MATHRIX VISION: DERİN TEŞHİS VE FARMASÖTİK ANALİZ")
st.write("---")

engine = load_mathrix_engine()
files = st.file_uploader("Mathrix Veri Girişi (Resimleri Sürükleyin)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files:
    if st.button("SİSTEMİ ÇALIŞTIR VE TEŞHİS KOY"):
        for file in files:
            img = Image.open(file).convert('RGB')
            
            # 1. Model Tahmini
            img_input = img.resize((224, 224))
            img_array = np.array(img_input) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if engine:
                preds = engine.predict(img_array)
                classes = ['Adenocarcinoma', 'Normal', 'Squamous Cell Carcinoma']
                res_key = classes[np.argmax(preds)]
                data = DRUG_DATABASE[res_key]
                
                # 2. Görüntü Üzerine Görsel Analiz Ekle
                analyzed_img = draw_analysis_overlay(img.copy(), data['label'], data['color'])
                
                # --- EKRAN TASARIMI ---
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(analyzed_img, caption="Mathrix Visual Scanning Output", use_container_width=True)
                
                with col2:
                    if res_key == 'Normal':
                        st.success(f"### {data['label']}")
                    else:
                        st.error(f"### {data['label']}")
                        st.markdown(f"*🔬 Önerilen Terapi:* {data['therapy']}")
                        st.markdown("#### 💊 Onaylı İlaç Protokolü:")
                        for drug in data['drugs']:
                            st.write(f"- {drug}")
                    
                    st.metric("Sistem Güven Endeksi", f"%{np.max(preds)*100:.2f}")
                st.write("---")
            else:
                st.error("Mathrix Core (lung_model.h5) yüklenemedi!")

# --- SIDEBAR ---
st.sidebar.header("SİSTEM MODÜLLERİ")
st.sidebar.write("🟢 Visual Analytics: ON")
st.sidebar.write("🟢 Drug Database: ON")
st.sidebar.write("🟢 Matrix Scannig: ON")
