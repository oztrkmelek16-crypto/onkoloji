import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Anonim Hücre Analiz Sistemi", layout="centered")

# Modeli Önbelleğe Al (Her seferinde yeniden yükleyip bilgisayarı yormasın)
@st.cache_resource
def load_prediction_model():
    try:
        # Yarın kaydettiğiniz modelin adı neyse buraya onu yazın
        model = load_model('lung_model.h5') 
        return model
    except Exception as e:
        return None

# --- ANALİZ FONKSİYONU ---
def process_and_predict(uploaded_file, model):
    # 1. Dosya ismini görmezden gel, doğrudan bitleri oku
    img_bytes = uploaded_file.getvalue() 
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # 2. Boyutlandırma (Modelin eğitim boyutuna göre 224 veya 180 yapın)
    img_resized = img.resize((224, 224))
    
    # 3. Sayısal Matrise Dönüştürme
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Normalizasyon (En kritik eksik buydu: 0-255 arasını 0-1 arasına çekiyoruz)
    img_array = img_array.astype('float32') / 255.0
    
    # 5. Tahmin
    predictions = model.predict(img_array)
    return predictions[0]

# --- ARAYÜZ ---
st.title("🔬 Hücre Tipi Analiz Laboratuvarı")
st.markdown("---")
st.write("*Melek - Akademik Veri Analiz Paneli*")

model = load_prediction_model()

if model is None:
    st.error("⚠️ 'lung_model.h5' bulunamadı! Lütfen model dosyasını ana dizine ekleyin.")
else:
    st.success("✅ Model başarıyla yüklendi. Analize hazır.")
    
    uploaded_file = st.file_uploader("Analiz edilecek hücre görüntüsünü seçin...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Anonimlik vurgusu
        st.info("🔄 Sistem: Dosya ismi gizlendi. Sadece piksel morfolojisi inceleniyor...")
        
        # Resmi göster
        st.image(uploaded_file, caption="Analiz Edilen Hücre Yapısı", use_container_width=True)
        
        if st.button("Hücreyi Teşhis Et"):
            with st.spinner('Yapay zeka katmanları analiz ediyor...'):
                probs = process_and_predict(uploaded_file, model)
                
                # Sınıflar (Sıralama modelin eğitim sırasıyla aynı olmalı!)
                classes = ['Adeno', 'Normal', 'Scc']
                max_idx = np.argmax(probs)
                result = classes[max_idx]
                confidence = probs[max_idx] * 100

                st.markdown("### 📊 Teşhis Sonucu")
                
                # Sonuç kartları
                c1, c2 = st.columns(2)
                c1.metric("Tespit Edilen Tip", result)
                c2.metric("Güven Oranı", f"%{confidence:.2f}")

                # Olasılık Çizelgesi
                st.bar_chart({classes[i]: float(probs[i]) for i in range(len(classes))})
                
                if confidence < 70:
                    st.warning("⚠️ Not: Güven oranı düşük. Veri kalitesini kontrol ediniz.")
                else:
                    st.balloons() # Başarıyı kutla!
