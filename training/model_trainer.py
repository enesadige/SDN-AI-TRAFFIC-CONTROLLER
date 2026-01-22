# -*- coding: utf-8 -*-
"""
Profesyonel Hibrit ML Modeli: Network Traffic Classification & Path Awareness
(REALISTIC & ANTI-LEAKAGE VERSION)
---------------------------------------------------------------------------
Bu sürümde 'Data Leakage' (Veri Sızıntısı) önlenmiştir.
Target'ı oluştururken kullanılan özet istatistikler (max_load vb.) eğitimden çıkarılmıştır.
Model, tıkanıklığı 'Sequence' verisinden kendisi öğrenmek zorundadır.

Mimari:
- Branch A (Scalar): Akış özellikleri (Hız, VIP vb.) -> Dense Network
- Branch B (Sequence): Yol durumları -> Transformer (Attention) Block
- Fusion: İki kolun birleşimi
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# AYARLAR
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data Collector'in urettigi dosya ismine dikkat edin
CSV_PATH = os.path.join(BASE_DIR, "../data/dataset_cleaned_final.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../models/hybrid_traffic_model.h5")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")
BATCH_SIZE = 64
EPOCHS = 60 
LEARNING_RATE = 0.0005 # Biraz daha yavas ve dikkatli ogrensin

# Hedef sınıfların anlamları (0, 1, 3 -> Mapped to 0, 1, 2)
TARGET_CLASSES = {0: "Normal", 1: "Congestion", 3: "Elephant"}

# ============================================================================
# VERİ ÖN İŞLEME (PREPROCESSING)
# ============================================================================

def parse_list_column(json_str):
    """CSV'deki '[0.1, 0.5]' şeklindeki stringleri listeye çevirir."""
    try:
        return json.loads(json_str)
    except:
        return []

def load_and_prep_data(csv_path):
    print(f"Veri yukleniyor: {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"UYARI: {csv_path} bulunamadi. Lutfen once network_monitor.py'yi calistirin.")
        # Test amaçlı dummy data oluşturabilir veya hata verebiliriz.
        raise FileNotFoundError(f"{csv_path} yok!")
        
    df = pd.read_csv(csv_path)
    
    # --- 1. Skaler Özellikler (X_scalar) ---
    # KRİTİK ADIM: DATA LEAKAGE ÖNLEME
    # Target'ı hesaplarken kullandığımız değişkenleri (max_load, avg_load, is_elephant)
    # modelin gözünden saklıyoruz. Model bunları Sequence verisinden kendisi çıkarmalı.
    
    leakage_cols = [
        'is_elephant', 'target', 
        'path_load_sorted', 'path_capacity_sorted', 'path_delay_sorted' # Sequence'a gidecekler  # KOPYA SÜTUNLAR (SİLİNDİ)
    ]
    
    # Sadece sayısal kolonları al ve yasaklıları düş
    scalar_cols = [c for c in df.columns if c not in leakage_cols]
    print(f"Skaler Ozellikler (Modelin Gordugu): {scalar_cols}")
    # Beklenen: ['flow_speed_mbps', 'is_vip', 'num_paths', 'min_path_capacity'...]
    
    X_scalar = df[scalar_cols].values
    
    # Skaler veriyi normalize et
    scaler = StandardScaler()
    X_scalar = scaler.fit_transform(X_scalar)
    
    # --- 2. Sıralı Özellikler (X_sequence) ---
    # Model, hangi yolun tıkalı olduğunu BURADAN öğrenecek.
    
    loads = df['path_load_sorted'].apply(parse_list_column).tolist()
    caps = df['path_capacity_sorted'].apply(parse_list_column).tolist()
    delays = df['path_delay_sorted'].apply(parse_list_column).tolist()
    
    max_paths = int(df['num_paths'].max())
    num_samples = len(df)
    
    # 3D Tensor: (Sample, Path, Features=[Load, Cap, Delay])
    X_sequence = np.zeros((num_samples, max_paths, 3), dtype=np.float32)
    
    print("Sequence verisi tensore donusturuluyor...")
    for i in range(num_samples):
        curr_paths = len(loads[i])
        for j in range(min(curr_paths, max_paths)):
            l = loads[i][j]         
            c = caps[i][j] / 1000.0 # Scale
            d = delays[i][j] / 100.0 # Scale
            
            X_sequence[i, j, 0] = l
            X_sequence[i, j, 1] = c
            X_sequence[i, j, 2] = d
            
    # --- 3. Hedef (y) ---
    y_raw = df['target'].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    print(f"Veri Hazir! Siniflar: {encoder.classes_}")
    
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    return X_scalar, X_sequence, y, max_paths, len(encoder.classes_)

# ============================================================================
# MODEL MİMARİSİ (REGULARIZED HYBRID)
# ============================================================================

def build_model(scalar_input_dim, max_paths, num_classes):
    # L2 Regularization ekleyerek ezberlemeyi zorlaştırıyoruz
    reg = regularizers.l2(0.001)

    # --- BRANCH 1: Skaler Girdi ---
    input_scalar = layers.Input(shape=(scalar_input_dim,), name="scalar_input")
    x1 = layers.Dense(64, activation="relu", kernel_regularizer=reg)(input_scalar)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.4)(x1) # Dropout artırıldı
    x1 = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x1)
    
    # --- BRANCH 2: Sequence Girdi (Yollar) ---
    input_seq = layers.Input(shape=(max_paths, 3), name="sequence_input")
    
    # Embedding benzeri projeksiyon
    x2 = layers.Dense(32, activation="relu")(input_seq)
    
    # ATTENTION: Model tıkanıklığı burada "sezmeli"
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x2, x2)
    x2 = layers.Add()([x2, attn_output]) 
    x2 = layers.LayerNormalization()(x2)
    
    # Global Pooling
    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dropout(0.3)(x2) # Dropout eklendi
    
    # --- FUSION ---
    concat = layers.Concatenate()([x1, x2])
    
    z = layers.Dense(64, activation="relu", kernel_regularizer=reg)(concat)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation="relu")(z)
    
    output = layers.Dense(num_classes, activation="softmax", name="prediction")(z)
    
    model = models.Model(inputs=[input_scalar, input_seq], outputs=output, name="Robust_Traffic_Classifier")
    
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    return model

# ============================================================================
# EĞİTİM
# ============================================================================

def train_and_evaluate():
    X_s, X_q, y, max_paths, num_classes = load_and_prep_data(CSV_PATH)
    
    # Stratify önemli: Sınıflar dengesiz olabilir
    X_s_train, X_s_test, X_q_train, X_q_test, y_train, y_test = train_test_split(
        X_s, X_q, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- CLASS WEIGHTS ---
    # 'Normal' sınıfı çok fazlaysa model tembellik edip hepsine 'Normal' diyebilir.
    # Bunu engellemek için azınlık sınıflarının ceza puanını artırıyoruz.
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Sinif Agirliklari: {class_weight_dict}")

    print("\n Model Insa Ediliyor...")
    model = build_model(X_s.shape[1], max_paths, num_classes)
    model.summary()
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]
    
    print("\n Egitim Basliyor...")
    history = model.fit(
        [X_s_train, X_q_train], y_train,
        validation_data=([X_s_test, X_q_test], y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=class_weight_dict, # Dengesizliği çözer
        verbose=1
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f" Model kaydedildi: {MODEL_SAVE_PATH}")
    
    # --- TEST ---
    print("\n TEST SONUCLARI:")
    y_pred_probs = model.predict([X_s_test, X_q_test])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Target isimlerini dinamik alalım (Map'e göre)
    # y=0 -> class 0 (muhtemelen Normal)
    # y=1 -> class 1 (muhtemelen Congestion)
    # y=2 -> class 3 (muhtemelen Elephant)
    target_names = [f"Class {i}" for i in range(num_classes)] 
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Grafik
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss (Daha düşük = Daha iyi)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy (1.0 olmamalı, 0.90 civarı ideal)')
    plt.legend()
    
    plt.savefig("training_results.png")
    print(" Grafik kaydedildi.")

if __name__ == "__main__":
    train_and_evaluate()