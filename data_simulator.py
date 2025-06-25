import pandas as pd
import numpy as np
import random

def generate_uroflow_data(num_samples=100):
    """
    Sentetik uroflow ve klinik not verileri üretir.
    Amacımız, farklı ürolojik durumları temsil eden veri paternleri oluşturmaktır.
    """
    data = []
    
    # Tanılar için olası etiketler (Basit bir sınıflandırma için)
    diagnoses = ["Normal", "Obstrüktif", "Disfonksiyonel"]
    
    for i in range(num_samples):
        # Rastgele bir tanı atayalım
        diagnosis = random.choice(diagnoses)
        
        qmax, qave, volume, flow_time, notes = 0, 0, 0, 0, ""

        if diagnosis == "Normal":
            # Normal akış paternleri için örnek değerler
            volume = random.randint(200, 500) # ml
            flow_time = random.randint(20, 45) # saniye
            qmax = round(volume / flow_time * random.uniform(1.5, 2.5), 2) # ml/s
            qave = round(volume / flow_time * random.uniform(0.8, 1.2), 2) # ml/s
            notes = random.choice([
                "Şikayeti yok. Normal işeme paterni.",
                "Periyodik kontrol. Akış iyi.",
                "Mesane boşaltımı tam.",
                "Belirgin bir sorun yok."
            ])

        elif diagnosis == "Obstrüktif":
            # Obstrüktif (tıkayıcı) akış paternleri için örnek değerler
            volume = random.randint(150, 400) # ml
            flow_time = random.randint(35, 80) # saniye (daha uzun)
            qmax = round(volume / flow_time * random.uniform(0.8, 1.5), 2) # ml/s (düşük pik)
            qave = round(volume / flow_time * random.uniform(0.5, 0.9), 2) # ml/s (düşük ortalama)
            notes = random.choice([
                "İşemede zorlanma var. Akış zayıf.",
                "Gece sık idrara çıkma. Sabahları zorlanıyor.",
                "Tamamen boşaltamama hissi. Kesik kesik işeme.",
                "Prostat büyümesi şüphesi. Ikınarak işeme."
            ])

        elif diagnosis == "Disfonksiyonel":
            # Disfonksiyonel (uyumsuzluk) akış paternleri için örnek değerler
            # Genellikle düzensiz, kesintili akış
            volume = random.randint(100, 350) # ml
            flow_time = random.randint(15, 60) # saniye (çok değişken)
            qmax = round(volume / flow_time * random.uniform(0.5, 1.8), 2) # Değişken pik
            qave = round(volume / flow_time * random.uniform(0.3, 1.0), 2) # Değişken ortalama
            notes = random.choice([
                "Mesane kaslarında istemsiz kasılma. Ani işeme isteği.",
                "İdrar tutamama epizodları. Sıkışma hissi.",
                "İşeme sırasında ağrı veya yanma. Yetersiz boşaltım.",
                "Nörolojik durumlar mevcut. Mesane kaslarında zayıflık."
            ])
            
        data.append([qmax, qave, volume, flow_time, notes, diagnosis])

    # Pandas DataFrame'e dönüştürüyoruz
    df = pd.DataFrame(data, columns=["Qmax", "Qave", "Volume", "FlowTime", "ClinicalNotes", "Diagnosis"])
    return df

if __name__ == "__main__":
    # Örnek olarak 100 veri noktası oluşturalım ve ilk 5'ini gösterelim
    simulated_data = generate_uroflow_data(num_samples=100)
    print("Oluşturulan Sentetik Verinin İlk 5 Satırı:")
    print(simulated_data.head())
    
    # İstersen bu veriyi bir CSV dosyasına kaydedebiliriz.
    # simulated_data.to_csv("simulated_uroflow_data.csv", index=False)
    # print("\nVeri 'simulated_uroflow_data.csv' dosyasına kaydedildi.")