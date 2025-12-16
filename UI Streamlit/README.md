# 🌽 Corn Leaf Disease Classifier

Aplikasi klasifikasi penyakit daun jagung menggunakan Machine Learning dengan antarmuka Streamlit yang modern dan responsif.

## 📋 Deskripsi

Aplikasi ini melakukan inferensi untuk mengklasifikasikan penyakit pada daun jagung berdasarkan gambar yang diupload. Model XGBoost telah dilatih menggunakan fitur hand-crafted (Fine, Coarse, DOR) pada Google Colab.

### Kategori Klasifikasi

| Kelas | Deskripsi |
|-------|-----------|
| 🌿 Daun Sehat | Daun dalam kondisi sehat tanpa tanda-tanda penyakit |
| 🍂 Daun Rusak | Daun mengalami kerusakan fisik atau mekanis |
| 🍂 Hawar Daun | Northern Leaf Blight, disebabkan jamur Exserohilum turcicum |
| 🍂 Karat Daun | Rust, disebabkan jamur Puccinia sorghi |

## 🔬 Pipeline Inferensi

```
Upload Gambar → Preprocessing → Segmentasi Otsu → Ekstraksi Fitur → Prediksi XGBoost
```

### Detail Pipeline:

1. **Preprocessing**
   - Resize ke 256×256 piksel
   - Konversi ke RGB
   - Normalisasi [0, 1]
   - Konversi ke Grayscale

2. **Segmentasi Otsu**
   - Gaussian Blur untuk noise reduction
   - Otsu Thresholding untuk segmentasi

3. **Ekstraksi Fitur (313 dimensi)**
   - Fine Texture (LBP rotation-invariant): 256 dimensi
   - Coarse Texture (Gradient Histogram): 32 dimensi
   - DOR (Directional Order Relation): 25 dimensi

4. **Klasifikasi**
   - XGBoost Classifier
   - Output: Prediksi kelas + Probabilitas

## 📁 Struktur Folder

```
corn-leaf-disease-classifier/
│
├── app.py                    # Aplikasi Streamlit utama
├── requirements.txt          # Dependencies
├── README.md                 # Dokumentasi
│
├── model/
│   └── xgb_best_model.pkl    # Model XGBoost terlatih
│
├── assets/
│   └── sample_images/        # Contoh gambar untuk testing
│       ├── sehat.jpg
│       ├── rusak.jpg
│       ├── hawar.jpg
│       └── karat.jpg
│
└── modules/
    ├── __init__.py           # Package initialization
    ├── preprocessing.py      # Fungsi preprocessing gambar
    ├── segmentation.py       # Fungsi segmentasi Otsu
    ├── feature_extraction.py # Ekstraksi fitur Fine, Coarse, DOR
    ├── pipeline.py           # Pipeline inferensi lengkap
    └── utils.py              # Konstanta dan helper functions
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Akses Aplikasi

Buka browser dan akses: `http://localhost:8501`

## 📸 Screenshot

*Screenshot aplikasi akan ditampilkan di sini*

## 📊 Informasi Model

- **Algoritma**: XGBoost Classifier
- **Fitur Input**: 313 dimensi
- **Output**: 4 kelas dengan probabilitas
- **Training**: Dilakukan di Google Colab dengan hyperparameter tuning

## 👨‍💻 Teknologi

- Python 3.x
- Streamlit
- OpenCV
- NumPy
- Scikit-learn
- XGBoost
- Pillow

## 📝 Catatan

- Model hanya melakukan **inferensi**, tidak ada training di aplikasi
- Pastikan gambar yang diupload adalah gambar daun jagung dengan kualitas yang baik
- Hasil terbaik didapat dengan gambar yang jelas dan tidak blur

---

**Tugas Besar Machine Learning** | Klasifikasi Penyakit Daun Jagung
