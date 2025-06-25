import pandas as pd
import numpy as np
import random
import os

# --- Akış Eğrisi Simülasyon Fonksiyonları ---
def generate_normal_flow_curve(volume, flow_time, num_points=100):
    """Normal, çan şekilli bir akış eğrisi simüle eder."""
    t = np.linspace(0, flow_time, num_points)
    
    peak_time = flow_time * random.uniform(0.4, 0.6)
    std_dev = flow_time * random.uniform(0.15, 0.25)
    flow_rate = np.exp(-((t - peak_time)**2) / (2 * std_dev**2))

    scaling_factor = volume / (np.trapz(flow_rate, t) + 1e-9)
    flow_rate = flow_rate * scaling_factor

    fade_in_out = np.sin(np.linspace(0, np.pi, num_points))
    flow_rate = flow_rate * fade_in_out * random.uniform(0.8, 1.2)

    noise = np.random.normal(0, 0.5, num_points)
    flow_rate = flow_rate + noise
    flow_rate[flow_rate < 0] = 0 
    
    flow_rate[0] = 0.0
    flow_rate[-1] = 0.0

    return list(flow_rate)

def generate_obstructive_flow_curve(volume, flow_time, num_points=100):
    """Obstrüktif, düzleşmiş ve uzamış bir akış eğrisi simüle eder."""
    t = np.linspace(0, flow_time, num_points)
    
    peak_time = flow_time * random.uniform(0.5, 0.8)
    std_dev = flow_time * random.uniform(0.25, 0.4)
    
    flow_rate = np.exp(-((t - peak_time)**2) / (2 * std_dev**2))
    scaling_factor = volume / (np.trapz(flow_rate, t) + 1e-9)
    flow_rate = flow_rate * scaling_factor * random.uniform(0.4, 0.8) # Genel akışı düşür
    
    fade_in_out = np.sin(np.linspace(0, np.pi, num_points))
    flow_rate = flow_rate * fade_in_out * random.uniform(0.9, 1.1)

    noise = np.random.normal(0, 0.3, num_points)
    flow_rate = flow_rate + noise
    flow_rate[flow_rate < 0] = 0
    
    flow_rate[0] = 0.0
    flow_rate[-1] = 0.0

    return list(flow_rate)

def generate_dysfunctional_flow_curve(volume, flow_time, num_points=100):
    """Disfonksiyonel, kesik kesik veya değişken akış eğrisi simüle eder."""
    t = np.linspace(0, flow_time, num_points)
    flow_rate = np.zeros_like(t)
    
    num_segments = random.randint(2, 4)
    current_time_pos = 0
    remaining_volume = volume
    
    for i in range(num_segments):
        if remaining_volume <= 0: break
        
        segment_duration = random.uniform(0.1, 0.3) * flow_time / num_segments
        
        gap_duration = random.uniform(0.05, 0.1) * flow_time / num_segments
        current_time_pos += gap_duration
        
        segment_volume = remaining_volume * random.uniform(0.3, 0.7)
        if i == num_segments - 1:
            segment_volume = remaining_volume
            
        peak_time_in_segment = segment_duration * random.uniform(0.4, 0.6)
        std_dev_segment = segment_duration * random.uniform(0.1, 0.2)
        
        segment_t = np.linspace(0, segment_duration, int(num_points * segment_duration / flow_time))
        if len(segment_t) == 0: continue
        
        segment_flow_base = np.exp(-((segment_t - peak_time_in_segment)**2) / (2 * std_dev_segment**2))
        
        if np.trapz(segment_flow_base, segment_t) > 0:
            segment_flow = segment_flow_base / np.trapz(segment_flow_base, segment_t) * segment_volume
        else:
            segment_flow = np.zeros_like(segment_t)

        start_idx = np.where(t >= current_time_pos)[0][0] if len(np.where(t >= current_time_pos)[0]) > 0 else len(t)
        end_idx = min(len(t), start_idx + len(segment_flow))
        
        if start_idx < end_idx:
            flow_rate[start_idx:end_idx] = segment_flow[:(end_idx - start_idx)]
        
        current_time_pos += segment_duration
        remaining_volume -= segment_volume

    noise = np.random.normal(0, 0.8, num_points)
    flow_rate = flow_rate + noise
    flow_rate[flow_rate < 0] = 0
    
    flow_rate[0] = 0.0
    flow_rate[-1] = 0.0

    return list(flow_rate)


# --- Ana Veri Üretme Fonksiyonu ---
def generate_uroflow_data(num_samples=100):
    """
    Sentetik uroflow parametreleri, klinik notlar, akış eğrileri ve hasta bilgileri üretir.
    """
    data = []
    diagnoses = ["Normal", "Obstrüktif", "Disfonksiyonel"]
    
    genders = ["Erkek", "Kadın"]
    
    first_names_male = ["Ahmet", "Mehmet", "Ali", "Can", "Burak", "Emre", "Deniz"]
    first_names_female = ["Ayşe", "Fatma", "Zeynep", "Elif", "Deniz", "Aslı", "Gamze"]
    last_names = ["Yılmaz", "Demir", "Çelik", "Şahin", "Koç", "Can", "Arslan"]
    
    for i in range(num_samples):
        diagnosis = random.choice(diagnoses)
        
        # Yeni eklenen hasta bilgileri
        patient_id = f"PID{i+1:04d}" # Örn: PID0001, PID0002
        age = random.randint(20, 80)
        gender = random.choice(genders)

        # İsim ve Soyisim seçimi
        first_name = random.choice(first_names_male if gender == "Erkek" else first_names_female)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        
        patient_info_text = f"Hasta Adı: {full_name} | Yaş: {age} | Cinsiyet: {gender}"
        if age > 50 and gender == "Erkek" and diagnosis == "Obstrüktif":
            patient_info_text += " | Şüpheli BPH geçmişi."
        elif gender == "Kadın" and diagnosis == "Disfonksiyonel":
            patient_info_text += " | Pelvik taban disfonksiyonu notları."


        qmax, qave, volume, flow_time, notes, flow_curve = 0, 0, 0, 0, "", []
        
        if diagnosis == "Normal":
            volume = random.randint(200, 500)
            flow_time = random.randint(15, 35)
            flow_curve = generate_normal_flow_curve(volume, flow_time)
            notes = random.choice([
                "Şikayeti yok. Normal işeme paterni.",
                "Periyodik kontrol. Akış iyi.",
                "Mesane boşaltımı tam.",
                "Belirgin bir sorun yok."
            ])

        elif diagnosis == "Obstrüktif":
            volume = random.randint(150, 450)
            flow_time = random.randint(30, 80)
            flow_curve = generate_obstructive_flow_curve(volume, flow_time)
            notes = random.choice([
                "İşemede zorlanma var. Akış zayıf.",
                "Gece sık idrara çıkma. Sabahları zorlanıyor.",
                "Tamamen boşaltamama hissi. Kesik kesik işeme.",
                "Prostat büyümesi şüphesi. Ikınarak işeme."
            ])

        elif diagnosis == "Disfonksiyonel":
            volume = random.randint(100, 400)
            flow_time = random.randint(20, 70)
            flow_curve = generate_dysfunctional_flow_curve(volume, flow_time)
            notes = random.choice([
                "Mesane kaslarında istemsiz kasılma. Ani işeme isteği.",
                "İdrar tutamama epizodları. Sıkışma hissi.",
                "İşeme sırasında ağrı veya yanma. Yetersiz boşaltım.",
                "Nörolojik durumlar mevcut. Mesane kaslarında zayıflık."
            ])
        
        flow_curve_arr = np.array(flow_curve)
        qmax = round(np.max(flow_curve_arr), 2)
        qave = round(np.mean(flow_curve_arr), 2)

        data.append([patient_id, first_name, last_name, age, gender, patient_info_text, qmax, qave, volume, flow_time, notes, flow_curve, diagnosis])

    df = pd.DataFrame(data, columns=["PatientID", "FirstName", "LastName", "Age", "Gender", "PatientInfo", "Qmax", "Qave", "Volume", "FlowTime", "ClinicalNotes", "FlowCurve", "Diagnosis"])
    return df

# Veri dosyasını global bir değişkende tutalım ki her fonksiyonda tekrar yüklemeyelim.
# Bu, uygulamanın performansı için daha iyidir.
_loaded_data_df = None

def save_data_to_csv(dataframe, filename="simulated_uroflow_data.csv"):
    """
    DataFrame'i belirtilen CSV dosyasına kaydeder.
    FlowCurve listesini string olarak kaydeder.
    """
    df_copy = dataframe.copy()
    df_copy['FlowCurve'] = df_copy['FlowCurve'].apply(lambda x: str(x)) 
    file_path = os.path.join(os.path.dirname(__file__), filename)
    df_copy.to_csv(file_path, index=False)
    print(f"Veri '{filename}' dosyasına kaydedildi.")

def load_data_from_csv(filename="simulated_uroflow_data.csv"):
    """
    Belirtilen CSV dosyasından veriyi yükler ve DataFrame olarak döndürür.
    FlowCurve sütununu tekrar listeye çevirir.
    """
    global _loaded_data_df
    if _loaded_data_df is not None:
        return _loaded_data_df # Zaten yüklüyse tekrar yükleme

    file_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['FlowCurve'] = df['FlowCurve'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        _loaded_data_df = df # Yüklenen veriyi sakla
        return df
    else:
        print(f"Uyarı: '{filename}' dosyası bulunamadı. Lütfen önce veri oluşturun.")
        return None

def get_patient_info_by_id(patient_id):
    """
    Simüle edilmiş veri setinden PatientID'ye göre hasta bilgilerini döndürür.
    """
    df = load_data_from_csv()
    if df is None:
        return None, "Veri seti yüklenemedi."
    
    # ID'ye göre ilk eşleşen satırı bul
    patient_row = df[df['PatientID'] == patient_id]
    
    if not patient_row.empty:
        # İlgili hasta için uroflow parametrelerini ve klinik notları da döndürelim
        info = patient_row.iloc[0] # İlk eşleşen satırı al
        return {
            "PatientID": info["PatientID"],
            "FirstName": info["FirstName"],
            "LastName": info["LastName"],
            "Age": info["Age"],
            "Gender": info["Gender"],
            "PatientInfo": info["PatientInfo"],
            "Qmax": info["Qmax"],
            "Qave": info["Qave"],
            "Volume": info["Volume"],
            "FlowTime": info["FlowTime"],
            "ClinicalNotes": info["ClinicalNotes"],
            "FlowCurve": info["FlowCurve"]
        }, None
    else:
        return None, f"'{patient_id}' ID'li hasta bulunamadı."


if __name__ == "__main__":
    # Bu blok, data_handler.py dosyası doğrudan çalıştırıldığında (test amaçlı) çalışır.
    print("Sentetik veri oluşturuluyor ve kaydediliyor (data_handler.py doğrudan çalıştırıldı)...")
    simulated_data = generate_uroflow_data(num_samples=200) 
    save_data_to_csv(simulated_data)
    
    loaded_data = load_data_from_csv()
    if loaded_data is not None:
        print("\nKaydedilen verinin ilk 5 satırı:")
        print(loaded_data.head())
        print("\nİlk hastanın akış eğrisi verisi (ilk 10 nokta):")
        print(loaded_data['FlowCurve'][0][:10])
        
        # Örnek hasta ID'si ile bilgi çekme testi
        test_patient_id = simulated_data['PatientID'].iloc[0]
        patient_info, error = get_patient_info_by_id(test_patient_id)
        if patient_info:
            print(f"\nTest Edilen Hasta Bilgileri ({test_patient_id}):")
            print(patient_info)
        else:
            print(f"\nHasta bilgisi çekme hatası: {error}")