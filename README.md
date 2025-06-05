
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

### 🔹 1. Deteksi Botol dengan YOLOv8 (Ultralytics)

* Menggunakan framework **Ultralytics YOLOv8** untuk deteksi objek.
* Model dilatih untuk mengenali lokasi botol dalam gambar dan memberikan bounding box serta label.
* Dokumentasi resmi: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

### 🔹 2. Klasifikasi Jenis Botol dengan MobileNetV2

* Setelah deteksi, area bounding box diekstrak dan diproses oleh model klasifikasi.
* **MobileNetV2** digunakan karena arsitekturnya ringan dan efisien untuk klasifikasi gambar.
* Model ini mampu membedakan lima jenis botol berdasarkan fitur visual.

### 🔹 3. Integrasi & Visualisasi dengan Streamlit

* Pipeline deteksi + klasifikasi diintegrasikan dan dibungkus dalam antarmuka pengguna berbasis web menggunakan **Streamlit**.
* Pengguna dapat mengunggah gambar, melihat hasil deteksi dan klasifikasi, dan mengunduh gambar hasil anotasi.

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

📦 **Model disimpan dalam format**:

* `.pt` → format asli dari Ultralytics
* `.onnx` → untuk deployment lintas platform

---


---

### 🧩 MobileNetV2 – Image Classification

| Metrik              | Nilai  |
| ------------------- | ------ |
| Accuracy (train)    | 0.9982 |
| Validation Accuracy | 0.9708 |
| Train Loss          | 0.0143 |
| Validation Loss     | 0.0922 |

**Classification Report:**

```
                 precision    recall  f1-score   support

   Beer Bottles       0.98      0.97      0.97      1000
Plastic Bottles       0.99      1.00      0.99      1000
    Soda Bottle       0.95      0.96      0.96      1000
   Water Bottle       0.98      0.98      0.98      1000
    Wine Bottle       0.96      0.95      0.96      1000

       accuracy                           0.97      5000
      macro avg       0.97      0.97      0.97      5000
   weighted avg       0.97      0.97      0.97      5000
```


📦 **Model disimpan dalam format**:

* `.keras` → format standar dari TensorFlow/Keras

---

## 🧰 Kebutuhan & Arsitektur Proyek

### 🔧 Tools & Teknologi

* Python (Jupyter Notebook / Colab / VS Code)
* **Ultralytics YOLOv8** untuk object detection
* **TensorFlow + Keras** untuk training klasifikasi
* **ONNX** sebagai format interoperabilitas model YOLO
* **Streamlit** untuk UI interaktif berbasis web
* **OpenCV / PIL** untuk preprocessing gambar dan manipulasi bounding box

### 🧱 Arsitektur Sistem

1. Upload gambar oleh pengguna melalui antarmuka Streamlit.
2. YOLOv8 mendeteksi objek botol dan menghasilkan bounding box.
3. Tiap area bounding box dipotong dan dikirim ke MobileNetV2 untuk klasifikasi.
4. Output final berupa gambar anotasi (bounding box + label jenis botol) ditampilkan kepada pengguna.

---

## ✅ Kesimpulan

Proyek ini berhasil membangun pipeline deteksi dan klasifikasi objek botol dengan akurasi dan performa yang sangat baik. Kombinasi antara YOLOv8 dan MobileNetV2 memungkinkan sistem ini untuk:

* Mendeteksi posisi botol secara akurat
* Mengidentifikasi jenis botol secara tepat
* Menyediakan output visual interaktif untuk pengguna
