import os
import pandas as pd
import numpy as np
import random
import time
from collections import deque # Sınırlı boyutlu liste için

import data_handler # data_handler'daki eğri simülasyon fonksiyonlarını buradan çağıracağız

class BluetoothUroflowSimulator:
    """
    Locum cihazından geliyormuş gibi sentetik uroflow verisi akışı simüle eder.
    Bluetooth cihaz tarama ve bağlanma fonksiyonlarını da içerir.
    """
    def __init__(self, num_points_per_curve=100):
        self.is_connected = False
        self.is_streaming = False
        self.current_patient_data = None # Simüle edilen aktif hastanın verisi
        self.flow_curve_data = None      # Akış eğrisi listesi
        self.num_points_per_curve = num_points_per_curve # Akış eğrisindeki nokta sayısı
        self.current_point_idx = 0       # Akış eğrisinde o anki indeks
        self.flow_time_sec = 0           # Akış süresi (saniye)
        self.stream_start_time = None    # Akışın başlama zamanı

        # Canlı olarak biriken akış hızı verileri
        # Bu deque, get_latest_data_packet tarafından doldurulur ve app.py tarafından kullanılır.
        self.live_flow_points = deque(maxlen=self.num_points_per_curve) 

        self.connected_device_name = "Yok" # Bağlı cihazın adını tutacak
        
        # Simüle edilmiş yakındaki Bluetooth cihazları listesi
        self.SIMULATED_NEARBY_DEVICES = [
            {"name": "Locum Uroflow V1", "address": "00:1A:2B:3C:4D:5E", "is_uroflow": True},
            {"name": "Akıllı Telefon (Ayşe)", "address": "F1:A2:B3:C4:D5:E6", "is_uroflow": False},
            {"name": "Kablosuz Kulaklık", "address": "AB:CD:EF:12:34:56", "is_uroflow": False},
            {"name": "Locum Uroflow Klinik", "address": "11:22:33:44:55:66", "is_uroflow": True},
            {"name": "Doktorun Tableti", "address": "77:88:99:AA:BB:CC", "is_uroflow": False},
        ]

        print("Bluetooth Uroflow Simülatörü başlatıldı.")

    def get_simulated_nearby_devices(self):
        """
        Yakındaki Bluetooth cihazlarını taramayı simüle eder.
        Gerçekte bu, Bluetooth API'lerini çağırırdı.
        """
        print("Simülatör: Yakındaki cihazlar taranıyor...")
        time.sleep(2) # Tarama gecikmesi simülasyonu
        random.shuffle(self.SIMULATED_NEARBY_DEVICES) # Her seferinde farklı sıralama
        return self.SIMULATED_NEARBY_DEVICES[:random.randint(3, len(self.SIMULATED_NEARBY_DEVICES))] # Rastgele sayıda cihaz bulmuş gibi yap

    def connect_to_device(self, device_name):
        """
        Belirtilen cihaz adına bağlanmayı simüle eder.
        """
        if self.is_connected:
            print(f"Simülatör: Zaten '{self.connected_device_name}' cihazına bağlısınız.")
            return True, "Zaten bağlı."
        
        print(f"Simülatör: '{device_name}' cihazına bağlanılıyor...")
        time.sleep(1.5) # Bağlantı gecikmesi simülasyonu

        found_device = next((d for d in self.SIMULATED_NEARBY_DEVICES if d["name"] == device_name), None)

        if found_device and found_device["is_uroflow"]:
            self.is_connected = True
            self.connected_device_name = device_name
            print(f"Simülatör: '{device_name}' cihazına başarıyla bağlanıldı.")
            return True, f"'{device_name}' cihazına başarıyla bağlanıldı."
        elif found_device and not found_device["is_uroflow"]:
            print(f"Simülatör: '{device_name}' bir Uroflow cihazı değil. Bağlantı başarısız.")
            return False, f"'{device_name}' bir Uroflow cihazı değil."
        else:
            print(f"Simülatör: '{device_name}' cihazı bulunamadı veya menzilde değil.")
            return False, f"'{device_name}' cihazı bulunamadı."

    def disconnect(self):
        """Bluetooth cihazından bağlantıyı kesmeyi simüle eder."""
        if self.is_streaming: # Akış aktifse önce durdur
            self.stop_streaming()
        print("Simülatör: Cihazdan bağlantı kesiliyor...")
        time.sleep(0.5)
        self.is_connected = False
        self.connected_device_name = "Yok"
        self.current_patient_data = None
        self.flow_curve_data = None
        self.live_flow_points.clear()
        print("Simülatör: Bağlantı kesildi.")

    def start_streaming(self, patient_id=None):
        """
        Veri akışını başlatmayı simüle eder.
        Belirli bir hasta ID'si verilirse, o hastanın verisini kullanır.
        Yoksa rastgele yeni bir hasta simüle eder.
        """
        if not self.is_connected:
            print("Simülatör: Bağlı değil, akış başlatılamaz.")
            return False, "Cihaza bağlı değilsiniz."

        if self.is_streaming:
            print("Simülatör: Akış zaten devam ediyor.")
            return True, "Akış zaten devam ediyor."

        print("Simülatör: Veri akışı başlatılıyor...")
        self.is_streaming = True
        self.current_point_idx = 0
        self.live_flow_points.clear() # Yeni akış için deque'yi temizle
        self.stream_start_time = time.time()

        if patient_id:
            # Belirli bir hasta için geçmiş veriyi yükleyelim
            patient_info, error = data_handler.get_patient_info_by_id(patient_id)
            if patient_info:
                self.current_patient_data = patient_info
                self.flow_curve_data = patient_info['FlowCurve']
                self.flow_time_sec = patient_info['FlowTime']
                print(f"Simülatör: Hasta ID {patient_id} için akış başlatıldı.")
            else:
                print(f"Simülatör: Hasta ID {patient_id} bulunamadı, rastgele yeni hasta oluşturuluyor.")
                self._generate_random_patient_for_stream()
        else:
            self._generate_random_patient_for_stream()

        print(f"Simülatör: Akış başlatıldı. Tahmini süre: {self.flow_time_sec} saniye.")
        return True, "Akış başlatıldı."

    def stop_streaming(self):
        """Veri akışını durdurmayı simüle eder."""
        if self.is_streaming:
            print("Simülatör: Veri akışı durduruluyor.")
            self.is_streaming = False
            print("Simülatör: Veri akışı durduruldu.")
            self.live_flow_points.clear() # Akış durunca biriken noktaları temizle
            return True
        return False

    def get_latest_data_packet(self):
        """
        Akıştan en son veri paketini (bir veya birkaç nokta) simüle eder.
        """
        if not self.is_streaming or self.flow_curve_data is None:
            return None, "Akış aktif değil veya veri yok."

        if self.current_point_idx < len(self.flow_curve_data):
            # Birim zaman diliminde akış hızını al
            flow_rate_at_point = self.flow_curve_data[self.current_point_idx]
            
            # Canlı akış noktalarını biriktir
            self.live_flow_points.append(flow_rate_at_point)

            # Anlık parametreleri hesaplayalım (birikmiş verilere göre)
            # Eğer akış süresi 0 ise sıfıra bölme hatasını önle
            current_flow_time_elapsed = (self.current_point_idx / self.num_points_per_curve) * self.flow_time_sec
            if len(self.live_flow_points) > 0 and current_flow_time_elapsed > 0:
                current_volume = np.trapz(list(self.live_flow_points), 
                                      np.linspace(0, current_flow_time_elapsed, len(self.live_flow_points)))
            else:
                current_volume = 0.0 # Hacim yoksa 0

            current_qmax = np.max(list(self.live_flow_points))
            current_qave = np.mean(list(self.live_flow_points)) if len(self.live_flow_points) > 0 else 0
            
            packet = {
                "FlowRate": flow_rate_at_point,
                "CurrentQmax": round(current_qmax, 2),
                "CurrentQave": round(current_qave, 2),
                "CurrentVolume": round(current_volume, 2),
                "CurrentFlowTime": round(current_flow_time_elapsed, 2),
                "LiveFlowCurve": list(self.live_flow_points), # Biriken eğri verisi
                "PatientID": self.current_patient_data.get('PatientID', 'N/A') # Canlı akış hastasının ID'si
            }
            self.current_point_idx += 1
            return packet, None
        else:
            self.stop_streaming() # Eğri bittiğinde akışı durdur
            return None, "Akış tamamlandı."

    def _generate_random_patient_for_stream(self):
        """Rastgele yeni bir hasta verisi oluşturur ve akış için hazırlar."""
        df_single_patient = data_handler.generate_uroflow_data(num_samples=1)
        single_patient_info = df_single_patient.iloc[0].to_dict()
        
        self.current_patient_data = single_patient_info
        self.flow_curve_data = single_patient_info['FlowCurve']
        self.flow_time_sec = single_patient_info['FlowTime']
        print(f"Simülatör: Yeni rastgele hasta {single_patient_info['PatientID']} için akış başlatıldı.")

# Modül doğrudan çalıştırıldığında simülatörü test et
if __name__ == "__main__":
    # Bu blok, bluetooth_simulator.py dosyası doğrudan çalıştırıldığında (test amaçlı) çalışır.
    
    # data_handler.py'yi hiç çalıştırmadıysak veya CSV güncel değilse, burada veriyi oluşturalım.
    data_file_path = os.path.join(os.path.dirname(__file__), "simulated_uroflow_data.csv")
    if not os.path.exists(data_file_path):
        print("simulated_uroflow_data.csv bulunamadı, yeniden oluşturuluyor...")
        simulated_df = data_handler.generate_uroflow_data(num_samples=10) # Az örnekle test et
        data_handler.save_data_to_csv(simulated_df)
        print("Sentetik veri oluşturuldu.")
    else: # Eğer dosya varsa ama yeni PatientID/FirstName/FlowCurve sütunları eksikse yeniden oluştur
        try:
            temp_df = pd.read_csv(data_file_path)
            required_cols = ["PatientID", "FirstName", "FlowCurve"]
            if not all(col in temp_df.columns for col in required_cols):
                raise ValueError("CSV dosyası güncel değil veya eksik sütunlar içeriyor.")
            if "FlowCurve" in temp_df.columns and not temp_df['FlowCurve'].empty and isinstance(temp_df['FlowCurve'].iloc[0], str):
                eval(temp_df['FlowCurve'].iloc[0]) # Eval hatası kontrolü
        except (pd.errors.EmptyDataError, ValueError, SyntaxError, NameError, IndexError) as e:
            print(f"Mevcut veri dosyası güncel değil veya bozuk ({e}), yeniden oluşturuluyor...")
            simulated_df = data_handler.generate_uroflow_data(num_samples=10)
            data_handler.save_data_to_csv(simulated_df)
            print("Sentetik veri yeniden oluşturuldu.")
    
    simulator = BluetoothUroflowSimulator()
    print("\n--- Cihaz Tarama Testi ---")
    devices = simulator.get_simulated_nearby_devices()
    for d in devices:
        print(f"Bulunan Cihaz: {d['name']} ({d['address']})")

    if devices:
        test_device = next((d["name"] for d in devices if d["is_uroflow"]), None)
        if test_device:
            print(f"\n--- Cihaza Bağlanma Testi: {test_device} ---")
            success, msg = simulator.connect_to_device(test_device)
            print(f"Bağlantı Sonucu: {success}, Mesaj: {msg}")

            if success:
                print(f"\n--- Akış Başlatma Testi (bağlı cihazdan) ---")
                success, msg = simulator.start_streaming()
                print(f"Akış Başlatma Sonucu: {success}, Mesaj: {msg}")

                if success:
                    print("\nCanlı Akış Simülasyonu Başladı (Test):")
                    while simulator.is_streaming:
                        packet, status = simulator.get_latest_data_packet()
                        if packet:
                            print(f"Zaman: {packet['CurrentFlowTime']:.2f}s, Akış Hızı: {packet['FlowRate']:.2f} ml/s, "
                                  f"Qmax: {packet['CurrentQmax']:.2f}, Volume: {packet['CurrentVolume']:.2f}")
                            time.sleep(0.1) # Gerçek zamanlı akışı simüle etmek için küçük bir gecikme
                        else:
                            print(f"Akış Durumu: {status}")
                            break
                simulator.disconnect()
        else:
            print("\nHiç Uroflow cihazı bulunamadı.")
    else:
        print("\nHiç cihaz bulunamadı.")