
# 🧠 Computer Vision Project: Deteksi & Klasifikasi Jenis Botol Menggunakan YOLOv8 dan MobileNetV2

## 📌 Deskripsi Singkat

Proyek ini menggabungkan dua pendekatan Computer Vision utama, yaitu **YOLOv8** untuk *object detection* dan **MobileNetV2** untuk *image classification*, guna membangun sistem yang mampu **mendeteksi keberadaan botol dan mengklasifikasikan jenisnya secara otomatis**. Sistem ini dirancang untuk diterapkan secara interaktif melalui platform **Streamlit**.

---

## 🎯 Tujuan

* Membangun sistem deteksi dan klasifikasi gambar botol secara real-time.
* Menggabungkan dua jenis model deep learning (YOLO dan CNN) dalam satu alur kerja.
* Menyediakan visualisasi hasil deteksi dan klasifikasi melalui antarmuka Streamlit yang ramah pengguna.
* Menguji dan mengevaluasi performa masing-masing model sebelum integrasi ke dalam aplikasi.

---

## 📁 Dataset

Dataset yang digunakan berasal dari:
[https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset](https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset)

Dataset ini berisi gambar botol sintetis dengan berbagai jenis, yaitu:

* **Beer Bottles**
* **Plastic Bottles**
* **Soda Bottles**
* **Water Bottles**
* **Wine Bottles**

Dataset dibagi ke dalam dua bagian:

* `dataset_train`: untuk pelatihan model klasifikasi
* `dataset_test`: untuk pengujian performa model

---

## 🧭 Alur dan Perencanaan Proyek

### 🔹 1. Deteksi Botol dengan YOLOv8

* Menggunakan YOLOv8 untuk mendeteksi keberadaan botol dalam gambar.
* Hasil deteksi berupa bounding box dan label posisi objek botol.

### 🔹 2. Klasifikasi Jenis Botol dengan MobileNetV2

* Setelah dideteksi, setiap instance gambar botol dipotong dari hasil deteksi dan diklasifikasikan menggunakan model CNN MobileNetV2.
* Model ini mengidentifikasi jenis botol berdasarkan fitur visual.

### 🔹 3. Integrasi dan Visualisasi

* Kedua model diintegrasikan dalam pipeline Python untuk menghasilkan prediksi terpadu.
* Pipeline ini akan diimplementasikan dalam antarmuka **Streamlit** untuk memudahkan pengguna mengunggah gambar, melihat hasil deteksi-klasifikasi, dan mengunduh hasil jika diperlukan.

---

## 📊 Evaluasi Model

### 🧩 YOLOv8 – Object Detection

| Metrik              | Nilai  |
| ------------------- | ------ |
| mAP\@0.5–0.95       | 0.9508 |
| mAP\@0.5            | 0.9947 |
| mAP\@0.75           | 0.9874 |
| Mean Precision (MP) | 0.9969 |
| Mean Recall (MR)    | 0.9803 |
| F1 Score (avg)      | 0.9885 |

🔧 **Model disimpan dalam format**:

* `yolo.pt` (format asli PyTorch)
* `yolo.onnx` (untuk kompatibilitas deployment)

---

### 🧩 MobileNetV2 – Image Classification

| Metrik              | Nilai      |
| ------------------- | ---------- |
| Accuracy (train)    | 1.0000     |
| Validation Accuracy | 0.9994     |
| Train Loss          | 1.0408e-04 |
| Validation Loss     | 0.0035     |

🔧 **Model disimpan dalam format**:

* `mobilenetv2-tuning.keras`

---

## 🧰 Kebutuhan & Arsitektur Proyek

### 🔧 Tools & Teknologi

* **Python** untuk scripting
* **YOLOv8** (Ultralytics) untuk object detection
* **TensorFlow + Keras** untuk training model MobileNetV2
* **ONNX** sebagai format interoperabilitas model YOLO
* **Streamlit** untuk UI interaktif berbasis web
* **OpenCV / PIL** untuk preprocessing gambar dan manipulasi bounding box

### 🧱 Arsitektur Sistem

1. Upload gambar melalui Streamlit
2. YOLOv8 mendeteksi botol dan menghasilkan bounding box
3. Gambar dalam bounding box dipotong lalu diklasifikasi oleh MobileNetV2
4. Hasil deteksi dan klasifikasi ditampilkan di layar beserta confidence score

---

## ✅ Kesimpulan

Proyek ini berhasil membangun pipeline deteksi dan klasifikasi objek botol dengan akurasi dan performa yang sangat baik. Kombinasi antara YOLOv8 dan MobileNetV2 memungkinkan sistem ini untuk:

* Mendeteksi posisi botol secara akurat
* Mengidentifikasi jenis botol secara tepat
* Menyediakan output visual interaktif untuk pengguna
