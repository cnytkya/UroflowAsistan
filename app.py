import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os
import nltk
import time
from collections import deque 

# Grafik çizimi için
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Kendi modüllerimizi içe aktarıyoruz
import data_handler
import ml_model_handler
import bluetooth_simulator

# Global değişkenler (Tüm fonksiyonlar tarafından erişilebilir olması için en başta tanımlanır)
app_widgets = {}
current_flow_curve_data = [] # Çizilecek akış eğrisi verisini tutacak (manuel/yüklü/canlı)
current_flow_time = 0 # Akış süresini de tutalım (manuel/yüklü/canlı)

# Animasyon kontrol değişkenleri
animation_running = False
animation_id = None # root.after metodu ID'sini tutar
animation_idx = 0 # Animasyonda o anki veri noktası indeksi

# Bluetooth Simülatör Instance'ı (Uygulama başlatıldığında create_main_window içinde oluşturulacak)
bluetooth_sim = None
live_stream_job_id = None # root.after döngüsünün ID'si


# --- Yardımcı Fonksiyonlar (En Temelden Başlayarak YUKARIDAN aşağıya doğru tanımlanır) ---

# 1. Renk hesaplama fonksiyonu (diğer fonksiyonlar kullanabilir)
def get_text_color_for_background(hex_color):
    """
    Verilen HEX arka plan rengine göre uygun bir metin rengi (siyah veya beyaz) döndürür.
    (WCAG 2.0 Kontrast Oranı algoritmasını basitleştirilmiş hali)
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        rgb = tuple(int(c*2, 16) for c in hex_color)
    else:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255

    if luminance > 0.5:
        return "black"
    else:
        return "white"

# 2. Canlı veri gösterim alanlarını temizleme fonksiyonu (canlı akış durduğunda kullanılır)
def clear_live_data_fields():
    """Canlı veri gösterim alanlarını temizler."""
    # Sadece widgetlar varsa temizle
    if "live_flow_rate_label" in app_widgets:
        app_widgets["live_flow_rate_label"].config(text="Anlık Akış Hızı: -")
        app_widgets["live_qmax_label"].config(text="Anlık Qmax: -")
        app_widgets["live_qave_label"].config(text="Anlık Qave: -")
        app_widgets["live_volume_label"].config(text="Anlık Hacim: -")
        app_widgets["live_flow_time_label"].config(text="Anlık Süre: -")
    
    # Grafiği de temizle (eğer canlı grafik figürü varsa)
    if "live_chart_figure" in app_widgets: 
        if app_widgets["live_chart_canvas_widget"].winfo_exists(): # Widget hala mevcutsa yok et
            app_widgets["live_chart_canvas_widget"].destroy()
        # matplotlib objelerini de silerek belleği boşalt
        if app_widgets["live_chart_figure"] in plt.get_fignums(): # Figür hala açık Matplotlib listesinde mi
            plt.close(app_widgets["live_chart_figure"]) # Kapat
        # Sözlükten referansları da kaldır
        del app_widgets["live_chart_figure"]
        del app_widgets["live_chart_ax"]
        del app_widgets["live_chart_line"]
        del app_widgets["live_chart_canvas"]
        del app_widgets["live_chart_canvas_widget"]

# 3. Manuel giriş alanlarının durumunu ayarlama fonksiyonu
def set_manual_input_state(state):
    """Manuel giriş alanlarının etkin/devre dışı durumunu ayarlar."""
    for entry_name in ["qmax_entry", "qave_entry", "volume_entry", "flow_time_entry"]:
        if entry_name in app_widgets: # Widget var mı kontrolü
            app_widgets[entry_name].config(state=state)
    if "notes_text_widget" in app_widgets:
        app_widgets["notes_text_widget"].config(state=state)
    if "analyze_button" in app_widgets:
        app_widgets["analyze_button"].config(state=state)
    if "load_patient_button" in app_widgets:
        app_widgets["load_patient_button"].config(state=state)
    if "patient_id_entry" in app_widgets:
        app_widgets["patient_id_entry"].config(state=state)
    
    # Radio butonların da durumunu ayarla
    # Bu widget'lar da app_widgets içinde var olmalı (create_main_window'da oluşturulacaklar)
    if "live_stream_radio" in app_widgets:
        app_widgets["live_stream_radio"].config(state=state)
    if "manual_input_radio" in app_widgets:
        app_widgets["manual_input_radio"].config(state=state)


# 4. Veri kaynağı modu değiştiğinde çağrılan fonksiyon (Radiobutton'dan)
# Bu fonksiyon, set_manual_input_state'i kullandığı için ondan sonra tanımlanır.
def on_data_source_change():
    mode = app_widgets["data_source_var"].get()
    if mode == "manual_input":
        set_manual_input_state(tk.NORMAL) # Manuel girişleri aktif et
    elif mode == "live_stream":
        set_manual_input_state(tk.DISABLED) # Canlı akış seçiliyse manuel girişleri kapat


# 5. Canlı akış döngüsünü durdurma fonksiyonu (bluetooth simülatörü kullanır)
def stop_live_stream_loop():
    """Tkinter after() döngüsünü durdurur."""
    global live_stream_job_id
    if live_stream_job_id:
        app_widgets["root"].after_cancel(live_stream_job_id)
        live_stream_job_id = None

# 6. Canlı akış eğrisini çizme fonksiyonu (periyodik olarak çağrılır)
def plot_live_flow_curve(flow_data, flow_time_current):
    """
    Canlı akış verisini matplotlib kullanarak çizer ve Tkinter penceresine gömer.
    Bu, reset_and_plot_static_curve'dan farklı olarak, her adımda yeniden çizilir.
    """
    # Sadece ilk kez figür oluştur veya temizlendiyse yeniden oluştur
    if "live_chart_figure" not in app_widgets:
        fig = plt.Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("Canlı Uroflow Akış Eğrisi")
        ax.set_xlabel("Zaman (s)")
        ax.set_ylabel("Akış Hızı (ml/s)")
        ax.grid(True)
        # Y ekseni limitini eğrinin toplam maksimumuna göre belirle (eğer biliniyorsa)
        ax.set_ylim(bottom=0, top=max(bluetooth_sim.flow_curve_data) * 1.2 if bluetooth_sim.flow_curve_data else 5)
        # X ekseni limitini toplam beklenen süreye göre belirle
        ax.set_xlim(0, bluetooth_sim.flow_time_sec * 1.05)
        
        canvas = FigureCanvasTkAgg(fig, master=app_widgets["chart_frame"])
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        app_widgets["live_chart_figure"] = fig
        app_widgets["live_chart_ax"] = ax
        app_widgets["live_chart_line"], = ax.plot([], [], color='blue', linewidth=2)
        app_widgets["live_chart_canvas"] = canvas
        app_widgets["live_chart_canvas_widget"] = canvas_widget
    
    # Veriyi güncelle
    ax = app_widgets["live_chart_ax"]
    line = app_widgets["live_chart_line"]
    canvas = app_widgets["live_chart_canvas"]

    time_points = np.linspace(0, flow_time_current, len(flow_data))
    line.set_data(time_points, flow_data)
    
    ax.set_xlim(0, flow_time_current * 1.05 if flow_time_current > 0 else bluetooth_sim.flow_time_sec * 1.05)

    canvas.draw_idle()


# 7. `reset_and_plot_static_curve` (Statik grafik çizimi, animasyon için kullanılır)
def reset_and_plot_static_curve():
    """
    Animasyonu sıfırlar ve akış eğrisinin tamamını tek bir statik grafikte gösterir.
    """
    global animation_running, animation_id, animation_idx

    # Animasyon çalışıyorsa durdur
    if animation_id:
        app_widgets["root"].after_cancel(animation_id)
        animation_id = None
    animation_running = False
    animation_idx = 0 # İndeksi sıfırla

    # Mevcut grafiği temizle (hem canlı hem de statik grafik için)
    if "chart_frame" in app_widgets and app_widgets["chart_frame"].winfo_children():
        for widget in app_widgets["chart_frame"].winfo_children():
            widget.destroy()
        if "live_chart_figure" in app_widgets: # Canlı grafik de kapatılmalı
            plt.close(app_widgets["live_chart_figure"]) 
            del app_widgets["live_chart_figure"]
            del app_widgets["live_chart_ax"]
            del app_widgets["live_chart_line"]
            del app_widgets["live_chart_canvas"]
            del app_widgets["live_chart_canvas_widget"]


    if not current_flow_curve_data: # Veri yoksa çizme
        return

    fig = plt.Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    
    time_points_full = np.linspace(0, current_flow_time, len(current_flow_curve_data)) # Tam zaman eksenini hesapla
    ax.plot(time_points_full, current_flow_curve_data, color='blue', linewidth=2)
    ax.set_title("Uroflow Akış Eğrisi")
    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("Akış Hızı (ml/s)")
    ax.grid(True) # Izgara ekle
    ax.set_ylim(bottom=0, top=max(current_flow_curve_data) * 1.2 if current_flow_curve_data else 5) # Y eksenini dinamik ayarla
    ax.set_xlim(0, current_flow_time * 1.05) # X eksenini de sabit tutalım

    if "chart_frame" in app_widgets and app_widgets["chart_frame"] is not None:
        canvas = FigureCanvasTkAgg(fig, master=app_widgets["chart_frame"])
        app_widgets["canvas_widget"] = canvas.get_tk_widget() # Canvas widget'ını saklayalım
        app_widgets["canvas_widget"].pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    else:
        print("Uyarı: chart_frame henüz oluşturulmamış, grafik çizilemedi.")

# 8. `stop_animation` (Animasyonu durdurma)
def stop_animation(show_message=False):
    """
    Grafik animasyonunu durdurur.
    """
    global animation_running, animation_id
    if animation_id:
        app_widgets["root"].after_cancel(animation_id)
        animation_id = None
    animation_running = False
    app_widgets["chart_summary_button"].config(text="Grafik Özeti (Oynat)")
    if show_message:
        messagebox.showinfo("Grafik Özeti", "Akış eğrisi oynatma tamamlandı.")
    pass

# 9. `animate_flow_curve` (Grafik özet animasyonunu oynatma)
def animate_flow_curve():
    """
    Akış eğrisini saniye saniye çizerek animasyon oluşturur.
    """
    global animation_running, animation_id, animation_idx

    if not current_flow_curve_data or current_flow_time <= 0: # Süre 0'dan büyük olmalı
        messagebox.showinfo("Grafik Özeti", "Analiz edilecek akış eğrisi verisi bulunamadı veya süre sıfır. Lütfen önce veri girip analiz edin.")
        return

    if animation_running:
        stop_animation()
        reset_and_plot_static_curve()
        return

    animation_running = True
    animation_idx = 0

    # Mevcut grafiği temizle (hem canlı hem de statik grafik için)
    if "chart_frame" in app_widgets and app_widgets["chart_frame"].winfo_children():
        for widget in app_widgets["chart_frame"].winfo_children():
            widget.destroy()
        if "live_chart_figure" in app_widgets:
            plt.close(app_widgets["live_chart_figure"]) 
            del app_widgets["live_chart_figure"]
            del app_widgets["live_chart_ax"]
            del app_widgets["live_chart_line"]
            del app_widgets["live_chart_canvas"]
            del app_widgets["live_chart_canvas_widget"]
    
    fig_anim = plt.Figure(figsize=(5, 3), dpi=100)
    ax_anim = fig_anim.add_subplot(111)
    ax_anim.set_title("Uroflow Akış Eğrisi (Oynatma)")
    ax_anim.set_xlabel("Zaman (s)")
    ax_anim.set_ylabel("Akış Hızı (ml/s)")
    ax_anim.grid(True)
    ax_anim.set_ylim(bottom=0, top=max(current_flow_curve_data) * 1.2 if current_flow_curve_data else 5)
    ax_anim.set_xlim(0, current_flow_time * 1.05)

    line, = ax_anim.plot([], [], color='blue', linewidth=2)
    
    canvas = FigureCanvasTkAgg(fig_anim, master=app_widgets["chart_frame"])
    app_widgets["canvas_widget"] = canvas.get_tk_widget()
    app_widgets["canvas_widget"].pack(fill=tk.BOTH, expand=True)
    canvas.draw()

    # Animasyonu güncelleme fonksiyonu
    def update_plot():
        global animation_idx, animation_id, animation_running
        
        if not animation_running:
            return

        time_points_current_segment = np.linspace(0, current_flow_time, len(current_flow_curve_data))[:animation_idx + 1]
        
        line.set_data(time_points_current_segment, current_flow_curve_data[:animation_idx + 1])
        canvas.draw()

        animation_idx += 1
        if animation_idx < len(current_flow_curve_data):
            delay_ms = int((current_flow_time / len(current_flow_curve_data)) * 1000) 
            if delay_ms == 0: delay_ms = 1
            
            animation_id = app_widgets["root"].after(delay_ms, update_plot)
        else:
            stop_animation(show_message=True)
            reset_and_plot_static_curve()

    app_widgets["chart_summary_button"].config(text="Grafik Özeti (Durdur)")
    update_plot()


# --- Ana İşlev Fonksiyonları (Düğmelere Bağlı Fonksiyonlar) ---
# Bu fonksiyonlar, yukarıdaki yardımcı fonksiyonları çağırır ve UI ile doğrudan etkileşim kurar.
# create_main_window'dan önce tanımlanmalıdır.

# 10. Bluetooth Cihaz Tarama fonksiyonu
def scan_for_devices():
    """Simüle edilmiş yakındaki Bluetooth cihazlarını tarar ve Combobox'a doldurur."""
    app_widgets["bluetooth_status_label"].config(text="Durum: Cihazlar Taranıyor...", foreground="orange")
    app_widgets["root"].update_idletasks() # UI'ı hemen güncelle
    
    # Gerçek Bluetooth taraması burada yapılır
    nearby_devices = bluetooth_sim.get_simulated_nearby_devices() # bluetooth_simulator'dan gerçek tarama çıktısı alıyor
    device_names = [d["name"] for d in nearby_devices]
    
    app_widgets["device_combobox"]['values'] = device_names
    if device_names:
        app_widgets["device_combobox"].set(device_names[0]) # İlk cihazı varsayılan seç
        app_widgets["connect_device_button"].config(state=tk.NORMAL) # Bağlan butonunu aktif et
        app_widgets["bluetooth_status_label"].config(text=f"Durum: {len(device_names)} cihaz bulundu.", foreground="blue")
    else:
        app_widgets["device_combobox"].set("")
        app_widgets["device_combobox"]['values'] = []
        app_widgets["connect_device_button"].config(state=tk.DISABLED)
        app_widgets["bluetooth_status_label"].config(text="Durum: Cihaz bulunamadı.", foreground="red")
    messagebox.showinfo("Bluetooth Tarama", f"{len(device_names)} cihaz bulundu. Listeden seçip bağlanabilirsiniz.")


# 11. Bluetooth Bağlanma fonksiyonu
def connect_to_selected_device():
    """Combobox'tan seçilen cihaza bağlanmayı simüle eder (gerçek bağlantı denemesi)."""
    selected_device_name = app_widgets["device_combobox"].get()
    if not selected_device_name:
        messagebox.showwarning("Uyarı", "Lütfen bağlanmak için bir cihaz seçin.")
        return

    if bluetooth_sim.is_connected:
        messagebox.showinfo("Bilgi", f"Zaten '{bluetooth_sim.connected_device_name}' cihazına bağlısınız.")
        return

    app_widgets["bluetooth_status_label"].config(text=f"Durum: '{selected_device_name}' bağlanılıyor...", foreground="orange")
    app_widgets["root"].update_idletasks() # UI'ı hemen güncelle
    
    # bluetooth_simulator.py'deki gerçek bağlantı metodunu çağır
    success, message = bluetooth_sim.connect_to_device(selected_device_name)

    if success:
        app_widgets["bluetooth_status_label"].config(text=f"Durum: Bağlı ({bluetooth_sim.connected_device_name})", foreground="green")
        app_widgets["connected_device_label"].config(text=f"Bağlı Cihaz: {bluetooth_sim.connected_device_name}") # Bağlı cihaz adını göster
        app_widgets["start_stream_button"].config(state=tk.NORMAL) # Akışı başlat butonunu aktif et
        app_widgets["disconnect_button"].config(state=tk.NORMAL) # Bağlantıyı kes butonunu aktif et
        app_widgets["connect_device_button"].config(state=tk.DISABLED) # Bağlanmayı devre dışı bırak
        app_widgets["scan_button"].config(state=tk.DISABLED) # Taramayı devre dışı bırak
        app_widgets["device_combobox"].config(state=tk.DISABLED) # Combobox'ı da devre dışı bırak
        messagebox.showinfo("Bağlantı Başarılı", message)
    else:
        app_widgets["bluetooth_status_label"].config(text=f"Durum: Bağlantı Hatası - {message}", foreground="red")
        app_widgets["connected_device_label"].config(text="Bağlı Cihaz: Yok")
        app_widgets["start_stream_button"].config(state=tk.DISABLED)
        messagebox.showerror("Bağlantı Hatası", message) # Hata mesajını daha belirgin göster

# 12. Bluetooth Bağlantıyı Kes fonksiyonu
def disconnect_bluetooth():
    """Bluetooth cihazından bağlantıyı kesmeyi simüle eder."""
    if not bluetooth_sim.is_connected:
        messagebox.showinfo("Bilgi", "Zaten bağlı değilsiniz.")
        return

    if bluetooth_sim.is_streaming: # Bağlantı kesmeden önce akışı durdur
        stop_live_stream_from_simulator() # Bu, start_stop_stream_button'ı da günceller
    
    bluetooth_sim.disconnect()
    app_widgets["bluetooth_status_label"].config(text="Durum: Bağlı Değil", foreground="red")
    app_widgets["connected_device_label"].config(text="Bağlı Cihaz: Yok")
    app_widgets["start_stream_button"].config(state=tk.DISABLED)
    app_widgets["stop_stream_button"].config(state=tk.DISABLED)
    app_widgets["disconnect_button"].config(state=tk.DISABLED)
    app_widgets["connect_device_button"].config(state=tk.NORMAL) # Bağlantı kesilince tekrar bağlantıya izin ver
    app_widgets["scan_button"].config(state=tk.NORMAL) # Taramayı tekrar aktif et
    app_widgets["device_combobox"].config(state=tk.NORMAL) # Combobox'ı tekrar aktif et
    app_widgets["device_combobox"].set("") # Combobox'ı temizle
    app_widgets["device_combobox"]['values'] = [] # Değerleri de temizle
    
    clear_live_data_fields() # Canlı veri gösterim alanlarını temizle
    messagebox.showinfo("Bağlantı Kesildi", "Bluetooth bağlantısı kesildi.")

# 13. Canlı Akışı Başlat fonksiyonu
def start_live_stream_from_simulator():
    """
    Simülatörden canlı veri akışını başlatır.
    """
    global live_stream_job_id, current_flow_curve_data, current_flow_time

    if not bluetooth_sim.is_connected:
        messagebox.showerror("Hata", "Bluetooth cihazına bağlı değilsiniz. Lütfen önce bağlanın.")
        return

    if bluetooth_sim.is_streaming:
        messagebox.showinfo("Bilgi", "Akış zaten devam ediyor.")
        return

    # Veri girişindeki PatientID'yi kullanarak belirli bir hastayı yükle
    patient_id_for_stream = app_widgets["patient_id_entry"].get().strip()
    
    success, message = bluetooth_sim.start_streaming(patient_id=patient_id_for_stream if patient_id_for_stream else None)
    
    if success:
        app_widgets["stream_status_label"].config(text="Akış Durumu: Aktif", foreground="green")
        app_widgets["start_stream_button"].config(state=tk.DISABLED)
        app_widgets["stop_stream_button"].config(state=tk.NORMAL)
        
        # UI'daki manuel giriş alanlarını devre dışı bırak
        set_manual_input_state(tk.DISABLED)
        # Radio butonları devre dışı bırak (akış varken mod değiştirilmesin)
        app_widgets["live_stream_radio"].config(state=tk.DISABLED)
        app_widgets["manual_input_radio"].config(state=tk.DISABLED)
        
        # Canlı veri akış döngüsünü başlat
        get_live_data_loop()
        messagebox.showinfo("Akış Başlatıldı", message)

        # Akış başladığında otomatik olarak Analiz Sonuçları sekmesine geç!
        app_widgets["notebook"].select(app_widgets["results_tab"]) 
    else:
        app_widgets["stream_status_label"].config(text=f"Durum: Akış Başlatılamadı ({message})", foreground="orange")
        messagebox.showerror("Akış Hatası", message)

# 14. Canlı Akışı Durdur fonksiyonu
def stop_live_stream_from_simulator():
    """
    Simülatörden canlı veri akışını durdurur.
    """
    global bluetooth_sim
    if bluetooth_sim.is_streaming:
        bluetooth_sim.stop_streaming()
        stop_live_stream_loop() 
        app_widgets["stream_status_label"].config(text="Akış Durumu: Durduruldu", foreground="red")
        app_widgets["start_stream_button"].config(state=tk.NORMAL)
        app_widgets["stop_stream_button"].config(state=tk.DISABLED)
        
        app_widgets["live_stream_radio"].config(state=tk.NORMAL) 
        app_widgets["manual_input_radio"].config(state=tk.NORMAL)
        
        final_packet_data, _ = bluetooth_sim.get_latest_data_packet() 
        if bluetooth_sim.current_patient_data and final_packet_data: 
            app_widgets["qmax_entry"].delete(0, tk.END)
            app_widgets["qmax_entry"].insert(0, str(final_packet_data['CurrentQmax']))
            app_widgets["qave_entry"].delete(0, tk.END)
            app_widgets["qave_entry"].insert(0, str(final_packet_data['CurrentQave']))
            app_widgets["volume_entry"].delete(0, tk.END)
            app_widgets["volume_entry"].insert(0, str(final_packet_data['CurrentVolume']))
            app_widgets["flow_time_entry"].delete(0, tk.END)
            app_widgets["flow_time_entry"].insert(0, str(final_packet_data['CurrentFlowTime']))
            app_widgets["notes_text_widget"].delete("1.0", tk.END)
            app_widgets["notes_text_widget"].insert("1.0", bluetooth_sim.current_patient_data['ClinicalNotes'])
            
            messagebox.showinfo("Akış Tamamlandı", "Canlı akış tamamlandı. Veriler giriş alanlarına aktarıldı, şimdi analiz edebilirsiniz.")
            app_widgets["notebook"].select(app_widgets["input_tab"]) 
        else:
            messagebox.showinfo("Akış Durduruldu", "Canlı akış durduruldu. Analiz için manuel veri girebilirsiniz.")
        
        on_data_source_change() 
    else:
        messagebox.showinfo("Bilgi", "Akış zaten aktif değil.")

# 15. Canlı veri akış döngüsünü çeken fonksiyon
def get_live_data_loop():
    """
    Tkinter'ın after() metodu ile periyodik olarak veri paketlerini çeker ve UI'ı günceller.
    """
    global live_stream_job_id, current_flow_curve_data, current_flow_time

    packet, status = bluetooth_sim.get_latest_data_packet()

    if packet:
        app_widgets["live_flow_rate_label"].config(text=f"Anlık Akış Hızı: {packet['FlowRate']:.2f} ml/s")
        app_widgets["live_qmax_label"].config(text=f"Anlık Qmax: {packet['CurrentQmax']:.2f} ml/s")
        app_widgets["live_qave_label"].config(text=f"Anlık Qave: {packet['CurrentQave']:.2f} ml/s")
        app_widgets["live_volume_label"].config(text=f"Anlık Hacim: {packet['CurrentVolume']:.2f} ml")
        app_widgets["live_flow_time_label"].config(text=f"Geçen Süre: {packet['CurrentFlowTime']:.2f} s")

        current_flow_curve_data = packet['LiveFlowCurve']
        current_flow_time = packet['CurrentFlowTime']
        
        plot_live_flow_curve(current_flow_curve_data, current_flow_time)

        live_stream_job_id = app_widgets["root"].after(100, get_live_data_loop)
    else:
        app_widgets["stream_status_label"].config(text=f"Akış Durumu: {status}", foreground="red")
        app_widgets["start_stream_button"].config(state=tk.NORMAL)
        app_widgets["stop_stream_button"].config(state=tk.DISABLED)
        
        stop_live_stream_loop() 
        
        pass


def load_patient_data(from_treeview=False):
    """
    Hasta Barkod ID'sine göre hasta bilgilerini yükler ve ilgili alanları doldurur.
    """
    patient_id = app_widgets["patient_id_entry"].get().strip()
    if not patient_id:
        if not from_treeview:
            messagebox.showerror("Hata", "Lütfen bir Hasta Barkod ID'si girin.")
        return

    if bluetooth_sim.is_streaming: 
        messagebox.showwarning("Uyarı", "Canlı akış devam ederken hasta yükleyemezsiniz. Lütfen akışı durdurun.")
        return

    print(f"DEBUG: Attempting to load patient info for ID: {patient_id}")
    patient_info, error = data_handler.get_patient_info_by_id(patient_id)
    print(f"DEBUG: Patient info load result: {'Success' if patient_info else 'Failure'}, Error: {error}")

    if patient_info:
        app_widgets["loaded_patient_info_label"].config(text=f"Yüklü Hasta: {patient_info['PatientInfo']}")
        
        app_widgets["data_source_var"].set("manual_input") # Manuel mod seçildi
        on_data_source_change() # UI durumunu güncelle

        app_widgets["qmax_entry"].delete(0, tk.END)
        app_widgets["qmax_entry"].insert(0, str(patient_info['Qmax']))
        app_widgets["qave_entry"].delete(0, tk.END)
        app_widgets["qave_entry"].insert(0, str(patient_info['Qave']))
        app_widgets["volume_entry"].delete(0, tk.END)
        app_widgets["volume_entry"].insert(0, str(patient_info['Volume']))
        app_widgets["flow_time_entry"].delete(0, tk.END)
        app_widgets["flow_time_entry"].insert(0, str(patient_info['FlowTime']))
        
        app_widgets["notes_text_widget"].delete("1.0", tk.END)
        app_widgets["notes_text_widget"].insert("1.0", patient_info['ClinicalNotes'])

        global current_flow_curve_data, current_flow_time
        current_flow_curve_data = patient_info['FlowCurve']
        current_flow_time = patient_info['FlowTime']
        
        reset_and_plot_static_curve()

        if not from_treeview:
            messagebox.showinfo("Başarılı", f"Hasta bilgileri '{patient_id}' yüklendi.")
            print("DEBUG: Info messagebox dismissed by user. Continuing with UI updates.") 
        
        app_widgets["notebook"].select(app_widgets["input_tab"]) 

        if not from_treeview:
            app_widgets["patients_tree"].selection_remove(app_widgets["patients_tree"].selection())
            app_widgets["patients_tree"].selection_remove(app_widgets["patients_tree"].selection())
            for item_id in app_widgets["patients_tree"].get_children():
                if app_widgets["patients_tree"].item(item_id, "values")[0] == patient_id:
                    app_widgets["patients_tree"].selection_add(item_id)
                    app_widgets["patients_tree"].focus(item_id)
                    app_widgets["patients_tree"].see(item_id)
                    break

    else:
        messagebox.showerror("Hata", f"Hasta bilgileri yüklenemedi: {error}")
        print("DEBUG: Error messagebox dismissed by user. Continuing with UI updates for error.")
        app_widgets["loaded_patient_info_label"].config(text="Yüklü Hasta: Yok")

def analyze_data():
    """
    'Analiz Et' düğmesine basıldığında çalışacak fonksiyon.
    Giriş kutularındaki ve metin alanındaki verileri alır, doğrular ve YZ modellerini çağırır.
    """
    global current_flow_curve_data, current_flow_time

    if bluetooth_sim.is_streaming: 
        messagebox.showinfo("Bilgi", "Canlı akış devam ederken analiz yapılamaz. Akışı durdurun.")
        return
    
    data_source_mode = app_widgets["data_source_var"].get()
    
    qmax, qave, volume, flow_time = 0.0, 0.0, 0.0, 0.0
    clinical_notes = ""
    analysis_flow_curve_data = []

    if data_source_mode == "live_stream" and len(bluetooth_sim.live_flow_points) > 0:
        qmax = np.max(list(bluetooth_sim.live_flow_points))
        qave = np.mean(list(bluetooth_sim.live_flow_points))
        volume = bluetooth_sim.current_patient_data['Volume'] if bluetooth_sim.current_patient_data else np.trapz(list(bluetooth_sim.live_flow_points), np.linspace(0, bluetooth_sim.flow_time_sec, len(bluetooth_sim.live_flow_points)))
        flow_time = bluetooth_sim.flow_time_sec
        clinical_notes = bluetooth_sim.current_patient_data['ClinicalNotes'] if bluetooth_sim.current_patient_data else ""
        analysis_flow_curve_data = list(bluetooth_sim.live_flow_points)
        
        messagebox.showinfo("Analiz Kaynağı", "Analiz, canlı akıştan gelen son veriler üzerinde yapılıyor.")

    else: 
        qmax_str = app_widgets["qmax_entry"].get()
        qave_str = app_widgets["qave_entry"].get()
        volume_str = app_widgets["volume_entry"].get()
        flow_time_str = app_widgets["flow_time_entry"].get()
        clinical_notes = app_widgets["notes_text_widget"].get("1.0", "end-1c").strip()

        try:
            qmax = float(qmax_str)
            qave = float(qave_str)
            volume = float(volume_str)
            flow_time = float(flow_time_str)
            
            if volume <= 0 or flow_time <= 0:
                raise ValueError("Hacim ve Süre pozitif olmalıdır.")

        except ValueError as e:
            messagebox.showerror("Giriş Hatası", f"Lütfen tüm sayısal alanlara geçerli pozitif bir sayı girin. ({e})")
            return

        if (not current_flow_curve_data or len(current_flow_curve_data) == 0 or 
            (current_flow_time != flow_time)):
            simulated_diagnosis_for_plot = "Normal" 
            if qmax < 10 or flow_time > 40:
                simulated_diagnosis_for_plot = "Obstrüktif"
            elif "ani işeme" in clinical_notes.lower() or "sıkışma" in clinical_notes.lower():
                simulated_diagnosis_for_plot = "Disfonksiyonel"

            if simulated_diagnosis_for_plot == "Normal":
                analysis_flow_curve_data = data_handler.generate_normal_flow_curve(volume, flow_time)
            elif simulated_diagnosis_for_plot == "Obstrüktif":
                analysis_flow_curve_data = data_handler.generate_obstructive_flow_curve(volume, flow_time)
            elif simulated_diagnosis_for_plot == "Disfonksiyonel":
                analysis_flow_curve_data = data_handler.generate_dysfunctional_flow_curve(volume, flow_time)
            
            current_flow_time = flow_time 
        else:
            analysis_flow_curve_data = current_flow_curve_data 

        messagebox.showinfo("Analiz Kaynağı", "Analiz, manuel giriş verileri üzerinde yapılıyor.")
    
    reset_and_plot_static_curve() 

    prediction_output = ml_model_handler.predict_uroflow_diagnosis(
        qmax, qave, volume, flow_time, clinical_notes, analysis_flow_curve_data
    )

    predicted_diagnosis = prediction_output["predicted_diagnosis"]
    text_analysis_info = prediction_output["text_analysis_info"]
    probabilities = prediction_output["probabilities"]

    result_text = (f"Model Tahmini: {predicted_diagnosis}\n"
                   f"Klinik Not Analizi: {text_analysis_info}\n"
                   f"Olasılıklar: {', '.join([f'{k}: {v}' for k, v in probabilities.items()])}")
                   
    app_widgets["result_label"].config(text=f"Analiz Sonucu: {result_text}")
    app_widgets["qmax_output_label"].config(text=f"Qmax: {qmax} ml/s")
    app_widgets["qave_output_label"].config(text=f"Qave: {qave} ml/s")
    app_widgets["volume_output_label"].config(text=f"Hacim: {volume} ml")
    app_widgets["flow_time_output_label"].config(text=f"Süre: {flow_time} s")
    app_widgets["notes_output_label"].config(text=f"Klinik Notlar: {clinical_notes[:100]}..." if clinical_notes else "Klinik Not Girilmedi.")

    messagebox.showinfo("Analiz Tamamlandı", "Veriler başarıyla analiz edildi. Sonuçlar 'Analiz Sonuçları' sekmesinde görüntülenebilir.")
    app_widgets["notebook"].select(app_widgets["results_tab"])

# 18. Hastaları listeleme fonksiyonu
def populate_patients_treeview():
    """
    Kayıtlı tüm sentetik hastaları Treeview'e yükler.
    """
    df = data_handler.load_data_from_csv()
    if df is None:
        messagebox.showerror("Hata", "Hasta verisi yüklenemedi.")
        return

    for item in app_widgets["patients_tree"].get_children():
        app_widgets["patients_tree"].delete(item)

    for index, row in df.iterrows():
        app_widgets["patients_tree"].insert("", tk.END, values=(
            row["PatientID"], 
            row["FirstName"],
            row["LastName"],
            row["Age"], 
            row["Gender"], 
            row["Diagnosis"],
            row["Qmax"],
            row["Qave"],
            row["Volume"],
            row["FlowTime"]
        ))

# 19. Hasta seçimi olayı fonksiyonu
def on_patient_select(event):
    """
    Treeview'de bir hasta seçildiğinde, PatientID'yi alıp load_patient_data'yı çağırır.
    """
    if bluetooth_sim.is_streaming: 
        messagebox.showwarning("Uyarı", "Canlı akış devam ederken hasta geçmişi yükleyemezsiniz. Lütfen akışı durdurun.")
        app_widgets["patients_tree"].selection_remove(app_widgets["patients_tree"].selection()) 
        return

    if not app_widgets["patients_tree"].selection():
        return

    selected_item = app_widgets["patients_tree"].selection()[0]
    patient_id = app_widgets["patients_tree"].item(selected_item, "values")[0]
    
    app_widgets["patient_id_entry"].delete(0, tk.END)
    app_widgets["patient_id_entry"].insert(0, patient_id)
    
    load_patient_data(from_treeview=True) 

# --- Ana Pencere Oluşturma Fonksiyonu (En Sonda ve Diğer Tüm Fonksiyonlar Tanımlandıktan Sonra) ---
def create_main_window():
    global app_widgets, bluetooth_sim 

    root = tk.Tk()
    app_widgets["root"] = root
    root.title("Uroflow Akıllı Asistanı")
    root.state('zoomed')
    root.resizable(True, True)

    # Bluetooth simülatörü objesini oluştur (root objesi oluşturulduktan sonra)
    bluetooth_sim = bluetooth_simulator.BluetoothUroflowSimulator()

    # --- İKON EKLEME ---
    icon_path = os.path.join(os.path.dirname(__file__), "app_icon.ico")
    if os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except tk.TclError:
            print(f"Uyarı: İkon yüklenemedi. İkon dosyası formatı desteklenmiyor veya bozuk: {icon_path}")
    else:
        print(f"Uyarı: İkon dosyası bulunamadı: {icon_path}")

    # --- Tema ve Stil Tanımlamaları ---
    style = ttk.Style(root)
    
    APP_BG_COLOR = "#E0F2F1" # Açık yeşil tonu (Mint Cream)
    
    root.configure(bg=APP_BG_COLOR)
    style.configure("TFrame", background=APP_BG_COLOR)
    style.configure("TLabel", 
                    font=("Helvetica", 10),
                    background=APP_BG_COLOR, 
                    foreground="#333333") 
    style.configure("Title.TLabel",
                    font=("Helvetica", 18, "bold"),
                    foreground="#2C3E50", 
                    background=APP_BG_COLOR) 

    # --- Buton Stili (Dinamik Yazı Rengi ile) ---
    BUTTON_BG_COLOR = "#4CAF50" # Düğme arka plan rengi (Koyu Yeşil)
    BUTTON_FG_COLOR = get_text_color_for_background(BUTTON_BG_COLOR)
    
    style.configure("TButton", 
                    font=("Helvetica", 12, "bold"),
                    background=BUTTON_BG_COLOR, 
                    foreground=BUTTON_FG_COLOR,    
                    padding=10)            
    
    BUTTON_BG_ACTIVE = "#45a049" # Fare üzerine gelince renk
    BUTTON_FG_ACTIVE = get_text_color_for_background(BUTTON_BG_ACTIVE)
    
    BUTTON_BG_PRESSED = "#367c39" # Tıklanınca renk
    BUTTON_FG_PRESSED = get_text_color_for_background(BUTTON_BG_PRESSED)

    style.map("TButton", 
              background=[('active', BUTTON_BG_ACTIVE), ('pressed', BUTTON_BG_PRESSED)], 
              foreground=[('active', BUTTON_FG_ACTIVE), ('pressed', BUTTON_FG_PRESSED)])

    # Sekme Stili
    style.configure("TNotebook", 
                    background=APP_BG_COLOR)
    style.configure("TNotebook.Tab", 
                    font=("Helvetica", 11, "bold"),
                    background="#BDC3C7", 
                    foreground="#34495E", 
                    padding=[10, 5])
    style.map("TNotebook.Tab", 
              background=[('selected', '#3498DB')], 
              foreground=[('selected', 'white')])

    # Treeview Stili (Hasta Geçmişi tablosu için)
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), background="#BDC3C7", foreground="#34495E") 
    style.configure("Treeview", font=("Helvetica", 9), rowheight=25, background="white", foreground="black", fieldbackground="white") 
    style.map("Treeview", 
              background=[('selected', '#3498DB')], 
              foreground=[('selected', 'white')])   


    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(expand=True, fill='both')

    app_widgets["notebook"] = ttk.Notebook(main_frame)
    app_widgets["notebook"].pack(expand=True, fill='both')

    # --- 1. Sekme: Veri Girişi ---
    input_tab = ttk.Frame(app_widgets["notebook"], padding="20")
    app_widgets["input_tab"] = input_tab
    app_widgets["notebook"].add(input_tab, text="Veri Girişi")

    input_tab.grid_rowconfigure(0, weight=0) # Title
    input_tab.grid_rowconfigure(1, weight=0) # Bluetooth Control Frame
    input_tab.grid_rowconfigure(2, weight=0) # Bluetooth Status Labels
    input_tab.grid_rowconfigure(3, weight=0) # Live Stream Control Frame
    input_tab.grid_rowconfigure(4, weight=0) # Live Data Display Frame
    input_tab.grid_rowconfigure(5, weight=0) # Patient ID Input Frame
    input_tab.grid_rowconfigure(6, weight=0) # Loaded Patient Info Label
    input_tab.grid_rowconfigure(7, weight=0) # Data Source Radio Buttons
    input_tab.grid_rowconfigure(8, weight=0) # Manual Input Section Label
    input_tab.grid_rowconfigure(9, weight=0) # Manual Input Frame (Qmax, Qave, etc.)
    input_tab.grid_rowconfigure(10, weight=0) # Notes label
    input_tab.grid_rowconfigure(11, weight=1) # Notes text area (dikeyde genişlesin)
    input_tab.grid_rowconfigure(12, weight=0) # Analyze Button
    input_tab.grid_columnconfigure(0, weight=1)
    input_tab.grid_columnconfigure(1, weight=1)


    title_label = ttk.Label(input_tab, text="Uroflow Veri Girişi", style="Title.TLabel")
    title_label.grid(row=0, column=0, columnspan=2, pady=10)

    # --- Bluetooth Kontrol Alanı (row 1) ---
    bluetooth_control_frame = ttk.Frame(input_tab, relief="groove", borderwidth=1, padding=10)
    bluetooth_control_frame.grid(row=1, column=0, columnspan=2, pady=(5,0), sticky="ew")
    bluetooth_control_frame.grid_columnconfigure(0, weight=1)
    bluetooth_control_frame.grid_columnconfigure(1, weight=1)
    bluetooth_control_frame.grid_columnconfigure(2, weight=1)
    bluetooth_control_frame.grid_columnconfigure(3, weight=1) 

    scan_button = ttk.Button(bluetooth_control_frame, text="Cihazları Tara", command=scan_for_devices, style="TButton")
    app_widgets["scan_button"] = scan_button
    scan_button.grid(row=0, column=0, padx=2, pady=5, sticky="ew")

    app_widgets["device_combobox"] = ttk.Combobox(bluetooth_control_frame, state="readonly", width=25)
    app_widgets["device_combobox"].grid(row=0, column=1, padx=2, pady=5, sticky="ew")

    connect_device_button = ttk.Button(bluetooth_control_frame, text="Bağlan", command=connect_to_selected_device, style="TButton", state=tk.DISABLED)
    app_widgets["connect_device_button"] = connect_device_button
    connect_device_button.grid(row=0, column=2, padx=2, pady=5, sticky="ew")

    disconnect_button = ttk.Button(bluetooth_control_frame, text="Bağlantıyı Kes", command=disconnect_bluetooth, style="TButton", state=tk.DISABLED)
    app_widgets["disconnect_button"] = disconnect_button
    disconnect_button.grid(row=0, column=3, padx=2, pady=5, sticky="ew")

    # Bluetooth Status (row 2)
    app_widgets["bluetooth_status_label"] = ttk.Label(input_tab, text="Durum: Bağlı Değil", foreground="red", font=("Helvetica", 10, "bold"))
    app_widgets["bluetooth_status_label"].grid(row=2, column=0, sticky="w", padx=5, pady=5)
    
    app_widgets["connected_device_label"] = ttk.Label(input_tab, text="Bağlı Cihaz: Yok", font=("Helvetica", 10, "italic"))
    app_widgets["connected_device_label"].grid(row=2, column=1, sticky="e", padx=5, pady=5)


    # --- Canlı Veri Akış Kontrolleri ve Gösterimi (row 3) ---
    live_stream_control_frame = ttk.Frame(input_tab, relief="groove", borderwidth=1, padding=10)
    live_stream_control_frame.grid(row=3, column=0, columnspan=2, pady=(10, 5), sticky="ew")
    live_stream_control_frame.grid_columnconfigure(0, weight=1)
    live_stream_control_frame.grid_columnconfigure(1, weight=1)
    live_stream_control_frame.grid_columnconfigure(2, weight=1) 

    start_stream_button = ttk.Button(live_stream_control_frame, text="Akışı Başlat", command=start_live_stream_from_simulator, style="TButton", state="disabled")
    app_widgets["start_stream_button"] = start_stream_button
    start_stream_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    stop_stream_button = ttk.Button(live_stream_control_frame, text="Akışı Durdur", command=stop_live_stream_from_simulator, style="TButton", state="disabled")
    app_widgets["stop_stream_button"] = stop_stream_button
    stop_stream_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    app_widgets["stream_status_label"] = ttk.Label(live_stream_control_frame, text="Akış Durumu: Durduruldu", foreground="red")
    app_widgets["stream_status_label"].grid(row=0, column=2, padx=5, pady=5, sticky="w")

    live_display_frame = ttk.Frame(input_tab) 
    live_display_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10), sticky="ew")
    live_display_frame.grid_columnconfigure(0, weight=1)
    live_display_frame.grid_columnconfigure(1, weight=1)

    app_widgets["live_flow_rate_label"] = ttk.Label(live_display_frame, text="Anlık Akış Hızı: -")
    app_widgets["live_flow_rate_label"].grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    app_widgets["live_qmax_label"] = ttk.Label(live_display_frame, text="Anlık Qmax: -")
    app_widgets["live_qmax_label"].grid(row=1, column=0, sticky="w", padx=5, pady=2)
    app_widgets["live_qave_label"] = ttk.Label(live_display_frame, text="Anlık Qave: -")
    app_widgets["live_qave_label"].grid(row=2, column=0, sticky="w", padx=5, pady=2)
    
    app_widgets["live_volume_label"] = ttk.Label(live_display_frame, text="Anlık Hacim: -")
    app_widgets["live_volume_label"].grid(row=1, column=1, sticky="w", padx=5, pady=2)
    app_widgets["live_flow_time_label"] = ttk.Label(live_display_frame, text="Geçen Süre: -")
    app_widgets["live_flow_time_label"].grid(row=2, column=1, sticky="w", padx=5, pady=2)


    # --- Hasta Barkod ID Girişi (row 5) ---
    patient_id_frame = ttk.Frame(input_tab)
    patient_id_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")
    patient_id_frame.grid_columnconfigure(0, weight=1)
    patient_id_frame.grid_columnconfigure(1, weight=1)
    patient_id_frame.grid_columnconfigure(2, weight=1)

    patient_id_label = ttk.Label(patient_id_frame, text="Hasta Barkod ID:")
    patient_id_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
    app_widgets["patient_id_entry"] = ttk.Entry(patient_id_frame, width=20)
    app_widgets["patient_id_entry"].grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    
    load_patient_button = ttk.Button(patient_id_frame, text="Bilgileri Yükle", command=lambda: load_patient_data(from_treeview=False), style="TButton")
    app_widgets["load_patient_button"] = load_patient_button
    load_patient_button.grid(row=0, column=2, padx=5, pady=5)

    app_widgets["loaded_patient_info_label"] = ttk.Label(input_tab, text="Yüklü Hasta: Yok", font=("Helvetica", 10, "italic"))
    app_widgets["loaded_patient_info_label"].grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)


    # --- Veri Kaynağı Seçimi (Radio Butonlar) ---
    # `data_source_var` ve `live_stream_radio`, `manual_input_radio` burada oluşturuluyor!
    # Bu yüzden on_data_source_change() ve set_manual_input_state() içinde bu widget'lar
    # kullanılmadan önce oluşturulmuş olmaları önemli.
    app_widgets["data_source_var"] = tk.StringVar(value="manual_input") # Varsayılan: Manuel Giriş

    data_source_radio_frame = ttk.Frame(input_tab, relief="groove", borderwidth=1, padding=10)
    data_source_radio_frame.grid(row=7, column=0, columnspan=2, pady=(15, 5), sticky="ew")
    app_widgets["data_source_radio_frame"] = data_source_radio_frame # Referansını sakla
    data_source_radio_frame.grid_columnconfigure(0, weight=1)
    data_source_radio_frame.grid_columnconfigure(1, weight=1)

    radio_label = ttk.Label(data_source_radio_frame, text="Veri Kaynağı:")
    radio_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

    live_stream_radio = ttk.Radiobutton(data_source_radio_frame, text="Canlı Akıştan", 
                                       variable=app_widgets["data_source_var"], 
                                       value="live_stream", command=on_data_source_change)
    live_stream_radio.grid(row=0, column=1, sticky="w", padx=5, pady=5)
    app_widgets["live_stream_radio"] = live_stream_radio

    manual_input_radio = ttk.Radiobutton(data_source_radio_frame, text="Manuel Giriş", 
                                        variable=app_widgets["data_source_var"], 
                                        value="manual_input", command=on_data_source_change)
    manual_input_radio.grid(row=0, column=2, sticky="w", padx=5, pady=5)
    app_widgets["manual_input_radio"] = manual_input_radio
    
    # --- Manuel Veri Girişi Alanları (şimdi 8. satırdan başlıyor) ---
    manual_input_section_label = ttk.Label(input_tab, text="Manuel Veri Girin:", font=("Helvetica", 12, "bold"))
    manual_input_section_label.grid(row=8, column=0, columnspan=2, pady=(15, 5), sticky="w") # Satır no güncellendi

    input_frame = ttk.Frame(input_tab)
    input_frame.grid(row=9, column=0, columnspan=2, pady=10, sticky="ew") # Satır no güncellendi
    input_frame.grid_columnconfigure(0, weight=1)
    input_frame.grid_columnconfigure(1, weight=1)

    labels = ["Maksimum Akış Hızı (ml/s):", "Ortalama Akış Hızı (ml/s):",
              "İşenen Hacim (ml):", "İşeme Süresi (s):"]
    entries = ["qmax_entry", "qave_entry", "volume_entry", "flow_time_entry"]

    for i, label_text in enumerate(labels):
        label = ttk.Label(input_frame, text=label_text)
        label.grid(row=i, column=0, sticky="w", padx=5, pady=5)
        entry = ttk.Entry(input_frame, width=20)
        entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
        app_widgets[entries[i]] = entry

    notes_label = ttk.Label(input_tab, text="Klinik Notlar/Semptomlar:")
    notes_label.grid(row=10, column=0, columnspan=2, pady=(10, 0)) # Satır no güncellendi

    app_widgets["notes_text_widget"] = tk.Text(input_tab, height=8, width=60, wrap='word', font=("Helvetica", 10), relief="solid", borderwidth=1, bg=APP_BG_COLOR) 
    app_widgets["notes_text_widget"].grid(row=11, column=0, columnspan=2, pady=5, sticky="nsew") # Satır no güncellendi

    analyze_button = ttk.Button(input_tab, text="Verileri Analiz Et", command=analyze_data, style="TButton")
    analyze_button.grid(row=12, column=0, columnspan=2, pady=20) # Satır no güncellendi
    app_widgets["analyze_button"] = analyze_button


    # --- 2. Sekme: Analiz Sonuçları ---
    results_tab = ttk.Frame(app_widgets["notebook"], padding="20")
    app_widgets["results_tab"] = results_tab
    app_widgets["notebook"].add(results_tab, text="Analiz Sonuçları")

    results_tab.grid_rowconfigure(0, weight=0)
    results_tab.grid_rowconfigure(1, weight=0)
    results_tab.grid_rowconfigure(2, weight=0)
    results_tab.grid_rowconfigure(3, weight=0)
    results_tab.grid_rowconfigure(4, weight=1) 
    results_tab.grid_rowconfigure(5, weight=3) 
    results_tab.grid_columnconfigure(0, weight=1)

    output_title_label = ttk.Label(results_tab, text="Analiz Sonuçları", style="Title.TLabel")
    output_title_label.grid(row=0, column=0, pady=10)

    app_widgets["result_label"] = ttk.Label(results_tab, text="Analiz Sonucu: Bekleniyor...", font=("Helvetica", 12, "italic"), wraplength=600, background=APP_BG_COLOR)
    app_widgets["result_label"].grid(row=1, column=0, pady=(5, 2), sticky="nsew")

    numeric_output_frame = ttk.Frame(results_tab)
    numeric_output_frame.grid(row=2, column=0, pady=(5,0), sticky="ew")
    numeric_output_frame.grid_columnconfigure(0, weight=1)
    numeric_output_frame.grid_columnconfigure(1, weight=1)

    app_widgets["qmax_output_label"] = ttk.Label(numeric_output_frame, text="Qmax: -")
    app_widgets["qmax_output_label"].grid(row=0, column=0, sticky="w", padx=10)

    app_widgets["qave_output_label"] = ttk.Label(numeric_output_frame, text="Qave: -")
    app_widgets["qave_output_label"].grid(row=1, column=0, sticky="w", padx=10)

    app_widgets["volume_output_label"] = ttk.Label(numeric_output_frame, text="Hacim: -")
    app_widgets["volume_output_label"].grid(row=0, column=1, sticky="w", padx=10)

    app_widgets["flow_time_output_label"] = ttk.Label(numeric_output_frame, text="Süre: -")
    app_widgets["flow_time_output_label"].grid(row=1, column=1, sticky="w", padx=10)
    
    app_widgets["notes_output_label"] = ttk.Label(results_tab, text="Klinik Notlar: -", wraplength=500, background=APP_BG_COLOR)
    app_widgets["notes_output_label"].grid(row=3, column=0, sticky="w", padx=10, pady=(5,5))

    chart_summary_button = ttk.Button(results_tab, text="Grafik Özeti (Oynat)", command=animate_flow_curve, style="TButton")
    app_widgets["chart_summary_button"] = chart_summary_button
    chart_summary_button.grid(row=4, column=0, pady=10)

    app_widgets["chart_frame"] = ttk.Frame(results_tab, borderwidth=2, relief="sunken")
    app_widgets["chart_frame"].grid(row=5, column=0, pady=10, sticky="nsew")

    # --- 3. Sekme: Hasta Geçmişi ---
    patients_history_tab = ttk.Frame(app_widgets["notebook"], padding="20")
    app_widgets["patients_history_tab"] = patients_history_tab
    app_widgets["notebook"].add(patients_history_tab, text="Hasta Geçmişi")

    patients_history_tab.grid_rowconfigure(0, weight=0) # Başlık
    patients_history_tab.grid_rowconfigure(1, weight=1) # Treeview (dikeyde genişlesin)
    patients_history_tab.grid_columnconfigure(0, weight=1) # Treeview (yatayda genişlesin)
    patients_history_tab.grid_columnconfigure(1, weight=0) # Kaydırma çubuğu sütunu

    history_title_label = ttk.Label(patients_history_tab, text="Kayıtlı Hasta Geçmişi", style="Title.TLabel")
    history_title_label.grid(row=0, column=0, columnspan=2, pady=10)

    columns = ("PatientID", "FirstName", "LastName", "Age", "Gender", "Diagnosis", "Qmax", "Qave", "Volume", "FlowTime")
    app_widgets["patients_tree"] = ttk.Treeview(patients_history_tab, columns=columns, show="headings")
    
    app_widgets["patients_tree"].heading("PatientID", text="Hasta ID")
    app_widgets["patients_tree"].heading("FirstName", text="Adı")
    app_widgets["patients_tree"].heading("LastName", text="Soyadı")
    app_widgets["patients_tree"].heading("Age", text="Yaş")
    app_widgets["patients_tree"].heading("Gender", text="Cinsiyet")
    app_widgets["patients_tree"].heading("Diagnosis", text="Tanı")
    app_widgets["patients_tree"].heading("Qmax", text="Qmax")
    app_widgets["patients_tree"].heading("Qave", text="Qave")
    app_widgets["patients_tree"].heading("Volume", text="Hacim")
    app_widgets["patients_tree"].heading("FlowTime", text="Süre")

    app_widgets["patients_tree"].column("PatientID", width=80, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("FirstName", width=100, anchor=tk.W)
    app_widgets["patients_tree"].column("LastName", width=100, anchor=tk.W)
    app_widgets["patients_tree"].column("Age", width=50, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("Gender", width=70, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("Diagnosis", width=100, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("Qmax", width=70, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("Qave", width=70, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("Volume", width=70, anchor=tk.CENTER)
    app_widgets["patients_tree"].column("FlowTime", width=70, anchor=tk.CENTER)

    app_widgets["patients_tree"].grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    scrollbar = ttk.Scrollbar(patients_history_tab, orient="vertical", command=app_widgets["patients_tree"].yview)
    scrollbar.grid(row=1, column=1, sticky="ns")
    app_widgets["patients_tree"].configure(yscrollcommand=scrollbar.set)

    app_widgets["patients_tree"].bind("<<TreeviewSelect>>", on_patient_select)
    
    # populate_patients_treeview() # Buradan kaldırıldı, en altta çağrılacak


    # --- 4. Sekme: Raporlar ---
    reports_tab = ttk.Frame(app_widgets["notebook"], padding="20")
    app_widgets["reports_tab"] = reports_tab
    app_widgets["notebook"].add(reports_tab, text="Raporlar")

    reports_tab.grid_rowconfigure(0, weight=0)
    reports_tab.grid_rowconfigure(1, weight=1)
    reports_tab.grid_columnconfigure(0, weight=1)

    reports_title_label = ttk.Label(reports_tab, text="Rapor Oluşturma ve Yönetimi", style="Title.TLabel")
    reports_title_label.grid(row=0, column=0, pady=10)

    report_info_label = ttk.Label(reports_tab, text="Bu bölümde analiz sonuçlarından raporlar oluşturabilir veya mevcut raporları dışa aktarabilirsiniz.", wraplength=700)
    report_info_label.grid(row=1, column=0, pady=10, sticky="n")
    
    export_report_button = ttk.Button(reports_tab, text="Mevcut Analiz İçin Rapor Oluştur (PDF)", command=lambda: messagebox.showinfo("Rapor", "PDF Rapor Oluşturma Fonksiyonu Buraya Gelecek!"), style="TButton")
    export_report_button.grid(row=2, column=0, pady=10)


    root.mainloop() # Tkinter olay döngüsünü başlatır

if __name__ == "__main__":
    # NLTK veri indirme ve veri/model hazırlığı
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        messagebox.showinfo("NLTK Veri İndirme", "NLTK veri setleri indiriliyor. Bu işlem biraz zaman alabilir.")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        messagebox.showinfo("NLTK Veri İndirme", "NLTK veri setleri başarıyla indirildi.")

    # Veri dosyasını kontrol et, yoksa oluştur ve kaydet.
    data_file_path = os.path.join(os.path.dirname(__file__), "simulated_uroflow_data.csv")
    if not os.path.exists(data_file_path):
        messagebox.showinfo("Veri Hazırlığı", "Uroflow verisi bulunamadı. Sentetik veri oluşturuluyor...")
        simulated_df = data_handler.generate_uroflow_data(num_samples=500)
        data_handler.save_data_to_csv(simulated_df, filename="simulated_uroflow_data.csv")
        messagebox.showinfo("Veri Hazırlığı", "Sentetik veri başarıyla oluşturuldu ve kaydedildi.")
    else:
        try:
            temp_df = pd.read_csv(data_file_path)
            required_cols = ["PatientID", "FirstName", "LastName", "FlowCurve"]
            if not all(col in temp_df.columns for col in required_cols):
                raise ValueError("CSV dosyası güncel değil veya eksik sütunlar içeriyor.")
            
            if "FlowCurve" in temp_df.columns and not temp_df['FlowCurve'].empty:
                for i in range(min(50, len(temp_df))): # İlk 50 satırı veya daha azını test et
                    test_val = temp_df['FlowCurve'].iloc[i]
                    if isinstance(test_val, str) and len(test_val.strip()) > 0:
                        eval(test_val) # Eval hatası verirse yakalanacak
            
        except (pd.errors.EmptyDataError, ValueError, SyntaxError, NameError, IndexError) as e:
            messagebox.showinfo("Veri Güncelleme", f"Mevcut veri dosyası güncel değil veya bozuk ({e}). Yeniden oluşturuluyor...")
            simulated_df = data_handler.generate_uroflow_data(num_samples=500)
            data_handler.save_data_to_csv(simulated_df, filename="simulated_uroflow_data.csv")
            messagebox.showinfo("Veri Güncelleme", "Sentetik veri başarıyla güncellendi ve kaydedildi.")


    if not ml_model_handler.load_models():
        messagebox.showinfo("Model Eğitimi", "Makine öğrenmesi modelleri eğitiliyor. Bu işlem biraz zaman alabilir.")
        ml_model_handler.train_and_save_models()
        messagebox.showinfo("Model Eğitimi", "Modeller başarıyla eğitildi ve kaydedildi.")
    
    # Tüm UI widget'ları create_main_window() içinde oluşturulduktan sonraki başlangıç ayarları.
    # root.mainloop()'tan hemen önce çağrılmalıdır.
    
    create_main_window() # UI'ı oluştur
    
    # UI oluştuktan sonra radio butonları ve diğer ilk ayarları yap
    # Bu satırlar, ilgili widget'lar create_main_window içinde oluşturulduktan sonra güvenle çağrılabilir.
    app_widgets["data_source_var"].set("manual_input") # Varsayılan: Manuel Giriş modunu seç
    on_data_source_change() # UI durumunu bu seçime göre ayarla (manuel girişleri aktif et)

    populate_patients_treeview() # Hasta Geçmişi tablosunu doldur
