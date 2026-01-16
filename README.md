# Discord Bot dengan AI dan Computer Vision

Bot Discord interaktif yang dilengkapi dengan AI generatif, klasifikasi gambar, dan deteksi objek menggunakan model-model machine learning yang telah dilatih.

## 📋 Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Arsitektur Bot](#arsitektur-bot)
- [Model yang Digunakan](#model-yang-digunakan)
- [Perintah-perintah Tersedia](#perintah-perintah-tersedia)
- [Cara Kerja AI](#cara-kerja-ai)

---

## 🤖 Fitur Utama

### 1. **AI Generatif (Chat Otomatis)**
Bot dapat membalas pesan secara otomatis menggunakan model bahasa AI tanpa memerlukan awalan perintah. Setiap pesan yang dikirim ke channel akan diproses dan menghasilkan balasan AI yang relevan.

### 2. **Klasifikasi Gambar (Image Classification)**
Menggunakan model Keras yang telah dilatih untuk mengklasifikasikan jenis burung (Pipit/Merpati) dari foto yang diunggah.

### 3. **Deteksi Objek (Object Detection)**
Mengidentifikasi dan menghitung semua objek dalam sebuah gambar menggunakan model YOLOv3.

### 4. **Perintah Utility**
Bot menyediakan berbagai perintah praktis seperti kalkulator, random image (anjing/bebek), password generator, dan file management.

---

## 🏗️ Arsitektur Bot

### Komponen Utama:

```
class_bot.py (Bot Utama)
├── AI Generative Response (TinyLlama Model)
├── Computer Vision
│   ├── Image Classification (Keras Model)
│   └── Object Detection (YOLOv3)
├── Utility Commands
└── File Management
```

### Dependensi Modul:
- `discord.py` - Library Discord API
- `transformers` - Untuk model AI (Hugging Face)
- `torch` - Framework deep learning
- `keras` - Untuk model klasifikasi
- `PIL` - Pemrosesan gambar
- `requests` - HTTP requests (untuk API gambar)

---

## 🧠 Model yang Digunakan

### 1. **TinyLlama-1.1B-Chat-v1.0** (AI Chat)
- **Fungsi**: Menghasilkan respons otomatis dalam percakapan
- **Ukuran**: 1.1 Miliar parameter (ringan dan cepat)
- **Device**: Otomatis menggunakan GPU (CUDA) jika tersedia, jika tidak menggunakan CPU
- **Proses**:
  1. Pesan user di-tokenize
  2. Model generate respons dengan max_length=100 token
  3. Respons dibersihkan dan dikirim ke Discord

### 2. **Keras Model** (Klasifikasi Burung)
- **File**: `keras_model.h5` + `labels.txt`
- **Input**: Gambar berukuran 224×224 pixel
- **Output**: Jenis burung + confidence score
- **Proses Normalisasi**:
  1. Gambar di-resize ke 224×224
  2. Nilai pixel dinormalisasi ke range [-1, 1]
  3. Prediksi menggunakan model
  4. Hasil dikembalikan dengan nama kelas dan akurasi

### 3. **YOLOv3** (Deteksi Objek)
- **File**: `yolov3.pt`
- **Fungsi**: Mendeteksi dan menghitung objek dalam gambar
- **Output**: Gambar dengan bounding box + daftar objek yang terdeteksi

---

## 💬 Perintah-perintah Tersedia

### **Perintah Matematika**
| Perintah | Contoh | Deskripsi |
|----------|--------|-----------|
| `+add` | `+add 5 3` | Penjumlahan (5 + 3 = 8) |
| `+min` | `+min 10 4` | Pengurangan (10 - 4 = 6) |
| `+times` | `+times 6 7` | Perkalian (6 × 7 = 42) |
| `+divide` | `+divide 20 4` | Pembagian (20 ÷ 4 = 5) |
| `+exp` | `+exp 2 8` | Pangkat (2^8 = 256) |

### **Perintah Hiburan**
| Perintah | Deskripsi |
|----------|-----------|
| `+dog` | Menampilkan gambar anjing acak |
| `+duck` | Menampilkan gambar bebek acak |
| `+coinflip` | Lempar koin (Head/Tail) |
| `+dice` | Lempar dadu (1-6) |
| `+meme` | Menampilkan meme dari folder lokal |
| `+hi` | Salam pembuka |
| `+bye` | Salam penutup |

### **Perintah File & Text**
| Perintah | Contoh | Deskripsi |
|----------|--------|-----------|
| `+tulis` | `+tulis Halo dunia` | Menyimpan teks ke file |
| `+tambahkan` | `+tambahkan Baris baru` | Menambah teks ke file |
| `+baca` | `+baca` | Membaca isi file |
| `+simpan` | `+simpan` (+ attachment) | Mengunduh file dari Discord |
| `+local_drive` | `+local_drive` | Melihat daftar file di folder |
| `+showfile` | `+showfile nama.txt` | Menampilkan file tertentu |

### **Perintah AI & Computer Vision**
| Perintah | Deskripsi |
|----------|-----------|
| `+klasifikasi` | Mengklasifikasikan jenis burung dari foto (Keras) |
| `+deteksi` | Mendeteksi semua objek dalam foto (YOLOv3) |

### **Perintah Lainnya**
| Perintah | Deskripsi |
|----------|-----------|
| `+pw` | Generate password random (10 karakter) |
| `+repeat` | Mengulangi pesan berkali-kali |
| `+waktu` | Menampilkan waktu sekarang |
| `+joined` | Menampilkan kapan member join server |

---

## 🔄 Cara Kerja AI

### **Respons Otomatis (Chat Mode)**
1. **Trigger**: Pesan dikirim tanpa awalan `+`
2. **Validasi**: Pesan tidak kosong dan < 1000 karakter
3. **Tokenisasi**: Pesan diubah menjadi token untuk model
4. **Generation**: Model menghasilkan respons dengan:
   - `max_length=100` (panjang maksimal respons)
   - `do_sample=False` (deterministic untuk kecepatan)
   - `num_beams=1` (tanpa beam search)
5. **Cleanup**: Respons dibersihkan dari special tokens
6. **Pengiriman**:
   - Jika < 2000 karakter: Dikirim langsung
   - Jika > 2000 karakter: Disimpan sebagai file `.txt`
7. **Error Handling**: Jika error, bot mengirim fallback message

### **Alur Eksekusi**
```
User Message
    ↓
on_message event
    ↓
Check if command (starts with '+')
    ↓
If YES → Process command
    ↓
If NO → Generate AI reply
    ↓
(Tokenize) → (Model Generate) → (Decode) → (Send)
```

---

## 📊 Performa Model

### TinyLlama pada CPU
- **Waktu Generation**: ~1-3 detik per respons
- **Memory**: ~2-3 GB RAM
- **Skalabilitas**: Cocok untuk deployment di resource terbatas

### Keras Model (Klasifikasi)
- **Waktu Inference**: < 1 detik
- **Akurasi**: Tergantung data training
- **Input**: Gambar JPG/PNG

### YOLOv3
- **Waktu Inference**: 1-2 detik
- **Deteksi**: 80+ kategori objek
- **Output**: Koordinat bounding box + confidence score

---

## ⚙️ Persyaratan Sistem

### **Minimum Requirements**
- Python 3.8+
- RAM: 4 GB (6+ recommended)
- Storage: 2-3 GB (untuk model-model)

### **GPU Support** (Opsional)
- NVIDIA GPU dengan CUDA support
- cuDNN library
- Otomatis dideteksi dan digunakan jika tersedia

---

## 🚀 Kesimpulan

Bot ini menggabungkan tiga teknologi utama:
1. **Generative AI** - Untuk percakapan natural
2. **Computer Vision** - Untuk analisis gambar
3. **Utility Functions** - Untuk fitur-fitur praktis

Semuanya terintegrasi dalam satu Discord bot yang responsif dan mudah digunakan, dengan error handling yang robust untuk memastikan bot tetap berjalan lancar meskipun ada masalah pada salah satu komponen.
