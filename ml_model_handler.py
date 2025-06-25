import pandas as pd
import numpy as np
import pickle
import os

# Makine Öğrenmesi için
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder # Tanı etiketlerini sayıya çevirmek için

# NLP için
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Kendi modülümüzü içe aktarıyoruz
import data_handler

# Modelleri kaydedeceğimiz/yükleyeceğimiz dizin
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# NLTK veri setlerini indir (eğer daha önce indirilmediyse)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK veri setleri indiriliyor...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    print("NLTK veri setleri indirildi.")


# --- NLP Ön İşleme Fonksiyonları ---
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Klinik notları NLP için ön işler: küçük harfe çevirme, noktalama kaldırma, tokenizasyon,
    durma kelimelerini çıkarma ve lemmatizasyon.
    """
    if not isinstance(text, str): # Metin olmayan girdileri stringe çevir
        return ""
        
    text = text.lower() # Küçük harfe çevir
    text = "".join([char for char in text if char.isalnum() or char.isspace()]) # Noktalama işaretlerini kaldır
    tokens = nltk.word_tokenize(text) # Kelimelere ayır (tokenizasyon)
    
    # Durma kelimelerini çıkar ve lemmatizasyon yap
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in nltk_stopwords
    ]
    return " ".join(processed_tokens) # Yeniden birleştir

# --- Akış Eğrisi Özellik Çıkarımı Fonksiyonları ---
def extract_flow_curve_features(flow_curve_data):
    """
    Uroflow akış eğrisinden önemli özellikleri çıkarır.
    Bu özellikler ML modeline girdi olarak kullanılacaktır.
    """
    if not flow_curve_data:
        return [0] * 5 # Boş ise varsayılan değerler döndür

    flow_curve_arr = np.array(flow_curve_data)
    
    peak_flow = np.max(flow_curve_arr)
    mean_flow = np.mean(flow_curve_arr)
    std_dev_flow = np.std(flow_curve_arr) 
    skewness_flow = pd.Series(flow_curve_arr).skew() 

    has_multiple_peaks = len(np.where(np.diff(np.sign(np.diff(flow_curve_arr))) < 0)[0]) > 1

    return [peak_flow, mean_flow, std_dev_flow, skewness_flow, int(has_multiple_peaks)]


# --- Global Model ve Vektörleyici Değişkenleri ---
combined_classifier = None # Hem sayısal hem metinsel hem de eğri özelliklerini kullanacak model
tfidf_vectorizer = None
label_encoder = None # Tanı etiketlerini sayıya çevirmek için

def train_and_save_models():
    """
    Sentetik veriyi kullanarak birleşik sınıflandırma modelini,
    TF-IDF vektörleyiciyi ve Label Encoder'ı eğitir ve diske kaydeder.
    """
    global combined_classifier, tfidf_vectorizer, label_encoder

    df = data_handler.load_data_from_csv()
    if df is None:
        print("Model eğitimi için veri bulunamadı. Lütfen önce veri oluşturun.")
        return False

    print("Modeller eğitiliyor...")

    # --- Özellik Mühendisliği ---
    # 1. Sayısal Özellikler
    numeric_features = ["Qmax", "Qave", "Volume", "FlowTime"]
    X_numeric = df[numeric_features] # X_numeric burada tanımlanıyor
    
    # 2. Metin Özellikleri (NLP Ön İşleme)
    df['ProcessedNotes'] = df['ClinicalNotes'].apply(preprocess_text)
    X_text = df['ProcessedNotes'] # X_text burada tanımlanıyor
    
    # 3. Akış Eğrisi Özellikleri
    flow_curve_features_df = pd.DataFrame(df['FlowCurve'].apply(extract_flow_curve_features).tolist(),
                                         columns=["PeakFlow_Curve", "MeanFlow_Curve", 
                                                  "StdDevFlow_Curve", "SkewnessFlow_Curve", "HasMultiplePeaks_Curve"])
    # Orijinal DataFrame ile birleştir (df'i güncelleyerek)
    df_combined_features = pd.concat([df, flow_curve_features_df], axis=1) # Yeni bir DataFrame oluşturmak daha güvenli


    # Tüm özellikleri birleştiriyoruz
    # X_features_for_combined: Sayısal ve eğri özelliklerini içeren DataFrame
    X_features_for_combined = df_combined_features[numeric_features + list(flow_curve_features_df.columns)] 
    
    # TF-IDF vektörleyiciyi eğit ve metin özelliklerini sayısal hale getir
    tfidf_vectorizer = TfidfVectorizer(max_features=500) 
    tfidf_vectorizer.fit(X_text) 
    X_text_tfidf = tfidf_vectorizer.transform(X_text) # Sparse matris
    
    # Sparse matrisi Dense DataFrame'e çevir
    X_text_tfidf_df = pd.DataFrame(X_text_tfidf.toarray(), 
                                   columns=[f"tfidf_{i}" for i in range(X_text_tfidf.shape[1])])
    
    # Sayısal ve metinsel özellikleri yatay olarak birleştir
    X_combined = pd.concat([X_features_for_combined, X_text_tfidf_df], axis=1) # X_combined burada tanımlanıyor

    # Hedef değişkeni (Diagnosis) sayısal etiketlere çevir
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["Diagnosis"]) # Diagnosis sütunu df içinde olmalı
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # --- Birleşik Sınıflandırıcı Eğitimi ---
    combined_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    combined_classifier.fit(X_train, y_train)

    # Model performansını değerlendir (sadece bilgi amaçlı)
    y_pred = combined_classifier.predict(X_test)
    print(f"Birleşik Model Doğruluğu: {accuracy_score(y_test, y_pred):.2f}")

    # Modelleri ve LabelEncoder'ı kaydet
    model_path = os.path.join(MODEL_DIR, "combined_classifier.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(combined_classifier, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"Modeller ve Label Encoder '{MODEL_DIR}' konumuna kaydedildi.")
    return True

def load_models():
    """
    Diskteki kayıtlı modelleri (birleşik sınıflandırıcı, TF-IDF vektörleyici, Label Encoder) yükler.
    """
    global combined_classifier, tfidf_vectorizer, label_encoder
    
    model_path = os.path.join(MODEL_DIR, "combined_classifier.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if (os.path.exists(model_path) and 
        os.path.exists(vectorizer_path) and 
        os.path.exists(encoder_path)):
        with open(model_path, 'rb') as f:
            combined_classifier = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Modeller başarıyla yüklendi.")
        return True
    else:
        print("Model dosyaları bulunamadı. Lütfen önce modelleri eğitin.")
        return False

def predict_uroflow_diagnosis(qmax, qave, volume, flow_time, clinical_notes, flow_curve_data):
    """
    Yeni bir hasta verisi için uroflow tanısını tahmin eder.
    Tüm veri türlerini (sayısal, metinsel, eğri) kullanır.
    """
    global combined_classifier, tfidf_vectorizer, label_encoder

    if combined_classifier is None or tfidf_vectorizer is None or label_encoder is None:
        print("Modeller yüklenmedi. Tahmin yapılamıyor.")
        return {
            "predicted_diagnosis": "Model Hatası: Modeller Yüklü Değil",
            "text_analysis_info": "Model Hatası",
            "probabilities": {}
        }

    # --- 1. Yeni veriden özellik çıkarma ---
    # Sayısal özellikler
    numeric_input_df = pd.DataFrame([[qmax, qave, volume, flow_time]], 
                                 columns=["Qmax", "Qave", "Volume", "FlowTime"])
    
    # Metinsel özellikler (NLP ön işleme)
    processed_note = preprocess_text(clinical_notes)
    text_vector = tfidf_vectorizer.transform([processed_note])
    text_input_df = pd.DataFrame(text_vector.toarray(), 
                                 columns=[f"tfidf_{i}" for i in range(text_vector.shape[1])])

    # Akış eğrisi özellikleri
    flow_curve_features = extract_flow_curve_features(flow_curve_data)
    flow_curve_input_df = pd.DataFrame([flow_curve_features],
                                       columns=["PeakFlow_Curve", "MeanFlow_Curve", 
                                                "StdDevFlow_Curve", "SkewnessFlow_Curve", "HasMultiplePeaks_Curve"])

    # Tüm özellikleri birleştir
    # predict_uroflow_diagnosis çağrılırken, tfidf_vectorizer'ın fit olduğu sütunları bekler.
    # Eğer tfidf_vectorizer 500 kelimeyle eğitildiyse, text_input_df'in de 500 sütunu olmalı.
    # Bunu sağlamanın en güvenli yolu, eğitilmiş tfidf_vectorizer'dan feature_names_in_ özelliğini kullanmaktır.
    
    # Tüm özellik sütunlarını oluşturmak için birleştirilmiş DataFrame'in sütunlarını al
    # Bu, eğitim setindeki sütun sıralamasına ve sayısına uyumu sağlar.
    
    # İlk olarak, tahmin için girdi olacak DataFrame'i manuel olarak oluşturalım,
    # ve eğitim setindeki sütunlara uygun hale getirelim.
    # Bu, train_and_save_models içinde X_combined'ın sütun adlarını bir yere kaydetmeyi gerektirebilir.
    # Şimdilik, sadece sırayı koruyarak birleştirme yapıyoruz.
    X_input_combined = pd.concat([numeric_input_df, flow_curve_input_df, text_input_df], axis=1)

    # ÖNEMLİ NOT: Tahmin yaparken, girdi DataFrame'inin sütunları, modelin eğitildiği
    # DataFrame'in sütunları ve sıralaması ile birebir aynı olmalıdır.
    # Bu projede, sütunların her zaman aynı sırayla eklendiğini varsaydık.
    # Daha sağlam bir sistemde, eğitilmiş modelle birlikte sütun adları da kaydedilip yüklenmelidir.

    # --- 2. Tahmin yapma ---
    predicted_encoded = combined_classifier.predict(X_input_combined)[0]
    predicted_diagnosis = label_encoder.inverse_transform([predicted_encoded])[0]

    # --- 3. Klinik not analizi için ek bilgi (manuel kontrol) ---
    text_analysis_info = ""
    if processed_note.strip() == "":
        text_analysis_info = "Klinik Not Girilmedi."
    elif "zorlanma" in processed_note or "zayif" in processed_note or "kesik" in processed_note:
        text_analysis_info = "Klinik Notlar Obstrüktif Belirtiler İçeriyor."
    elif "ani" in processed_note or "sikisma" in processed_note or "tutamama" in processed_note:
        text_analysis_info = "Klinik Notlar Disfonksiyonel Belirtiler İçeriyor."
    elif "normal" in processed_note or "sikayet yok" in processed_note or "iyi" in processed_note:
        text_analysis_info = "Klinik Notlar Normal Belirtiler İçeriyor."
    else:
        text_analysis_info = "Klinik Notlar Analiz Edildi."

    # Modelin güven skorunu da ekleyebiliriz (olasılıklar)
    prediction_probabilities = combined_classifier.predict_proba(X_input_combined)[0]
    prob_dict = {label: f"{prob:.2f}" for label, prob in zip(label_encoder.classes_, prediction_probabilities)}
    
    return {
        "predicted_diagnosis": predicted_diagnosis,
        "text_analysis_info": text_analysis_info,
        "probabilities": prob_dict
    }

if __name__ == "__main__":
    # Bu blok, ml_model_handler.py dosyası doğrudan çalıştırıldığında (test amaçlı) çalışır.
    print("ml_model_handler modülü çalıştırılıyor (Modeller Eğitiliyor ve Kaydediliyor)...")
    train_and_save_models()
    
    print("\nModeller yükleniyor...")
    load_models()
    
    # Örnek bir tahmin yapalım (gerçekçi değerler verilmeli)
    example_qmax = 10.0
    example_qave = 5.0
    example_volume = 250.0
    example_flow_time = 50.0
    example_notes = "İşemede çok zorlanıyorum, akışım kesik kesik."
    
    # Akış eğrisi verisini oluşturmak için data_handler'dan faydalanıyoruz
    example_curve = data_handler.generate_obstructive_flow_curve(example_volume, example_flow_time)
    
    prediction = predict_uroflow_diagnosis(example_qmax, example_qave, example_volume, example_flow_time, example_notes, example_curve)
    print("\nÖrnek Tahmin Sonucu:")
    print(f"Tahmini Tanı: {prediction['predicted_diagnosis']}")
    print(f"Metin Analizi: {prediction['text_analysis_info']}")
    print(f"Olasılıklar: {prediction['probabilities']}")