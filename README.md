# Flood Area Semantic Segmentation
### Deteksi Area Banjir dari Citra Udara dengan Deep Learning
**Mask2Former + Balanced Sampling + Lovász Loss** Tim Cungpret, NEXUS'26

---

## Daftar Isi
- [Gambaran Proyek](#gambaran-proyek)
- [Dataset](#dataset)
- [Pendekatan](#pendekatan)
- [Arsitektur Model](#arsitektur-model)
- [Hasil](#hasil)
- [Struktur Notebook](#struktur-notebook)
- [Cara Menjalankan](#cara-menjalankan)
- [Requirements](#requirements)
- [Struktur Output](#struktur-output)
- [Catatan Teknis](#catatan-teknis)

---

## Gambaran Proyek

Notebook ini menyajikan pipeline lengkap **segmentasi semantik area banjir** pada citra udara untuk kompetisi **NEXUS'26**. Setiap piksel pada citra aerial 640×480 diklasifikasikan ke dalam salah satu dari 10 kelas berikut:

| ID | Kelas | Kategori |
|---|---|---|
| 0 | `background` | Latar belakang |
| 1 | `building flooded` | Bangunan terendam **rare** |
| 2 | `building non-flooded` | Bangunan normal |
| 3 | `grass` | Vegetasi pendek |
| 4 | `pool` | Kolam renang **rare** |
| 5 | `road flooded` | Jalan terendam |
| 6 | `road non-flooded` | Jalan normal |
| 7 | `tree` | Pohon **rare** |
| 8 | `vehicle` | Kendaraan **rare** |
| 9 | `water` | Badan air |

**Tantangan utama**: severe class imbalance kelas `water` mendominasi >56% piksel, sementara `tree` dan `vehicle` masing-masing hanya <0,2%, menghasilkan **rasio imbalance 317,6:1**.

**Metrik evaluasi**: mIoU (mean Intersection over Union).

---

## Dataset

Dataset terdiri dari citra aerial RGB beserta mask anotasi (format Supervisely JSON dengan bitmap base64 per-objek), dibagi menjadi tiga split:

| Split | Isi | Path |
|---|---|---|
| Train | Gambar + mask | `dataset_640x480/train/{img,masks}` |
| Validation | Gambar + mask | `dataset_640x480/validation/{img,masks}` |
| Test | Gambar saja | `dataset_640x480/test/img` |

Seluruh gambar berukuran **640×480 piksel**. Selama training, gambar di-pad dan di-crop ke **512×512** agar kompatibel dengan input model transformer.

### Distribusi Kelas (Train)

| Kelas | Train px | Train % |
|---|---|---|
| water | 250.402.951 | 56,41% |
| road non-flooded | 77.509.681 | 17,46% |
| road flooded | 48.574.411 | 10,94% |
| pool | 24.491.705 | 5,52% |
| building non-flooded | 14.446.329 | 3,25% |
| grass | 12.199.829 | 2,75% |
| background | 7.644.786 | 1,72% |
| building flooded | 6.940.896 | 1,56% |
| vehicle | 905.084 | 0,20% |
| tree | 788.328 | 0,18% |

Pemindaian anomali label pada seluruh training set **tidak menemukan piksel di luar rentang kelas valid (0-9)**, mengonfirmasi integritas dataset.

---

## Pendekatan

| Komponen | Pilihan | Justifikasi |
|---|---|---|
| **Backbone** | Mask2Former Swin-Tiny | State-of-the-art untuk panoptic/semantic segmentation (CVPR 2022) |
| **Loss** | Lovász-Softmax + NLL hybrid | Lovász langsung mengoptimasi IoU; NLL untuk stabilitas awal training |
| **Sampling** | BalancedBatchSampler kustom | Menjamin kelas langka muncul di setiap batch, bukan hanya rata-rata per epoch |
| **Augmentasi** | Albumentations pipeline | Transform citra dan mask sinkron, lebih cepat dari torchvision |
| **Optimizer** | AdamW 3-grup learning rate | Backbone, decoder, dan head mendapat LR berbeda sesuai kematangan fitur |
| **Regularisasi** | EMA (decay 0,9999) + AMP + grad clip | Generalisasi lebih stabil, hemat memori, cegah gradient explosion |
| **Inference** | TTA horizontal flip + per-class threshold | Naikkan recall kelas minoritas tanpa retraining |

### Strategi Mengatasi Class Imbalance
Class imbalance ditangani di **tiga level berbeda** secara bersamaan:
1. **Loss level** `CLASS_WEIGHTS` memberi bobot lebih besar ke kelas rare (vehicle & tree = 8.0)
2. **Batch level** `BalancedBatchSampler` menjamin proporsi tetap (50% rare, 10% water, 40% normal) di setiap batch
3. **Inference level** `DEFAULT_CLASS_THRESHOLDS` menurunkan ambang keputusan untuk kelas rare (0,3) dibanding kelas dominan (0,5)

---

## Arsitektur Model

**Mask2FormerFlood** wrapper kustom di atas `Mask2FormerForUniversalSegmentation` (HuggingFace, pretrained `facebook/mask2former-swin-tiny-ade-semantic`, ADE20K 151 kelas):

- Jumlah query dikurangi dari 100 (default ADE20K) menjadi **20**, karena task ini hanya 10 kelas
- Head classifier dan query embedder direinisialisasi (`ignore_mismatched_sizes=True`) karena perbedaan jumlah kelas (151→10) dan jumlah query (100→20); bobot backbone Swin-Tiny dan decoder tetap memakai pretrained
- Output query-based (`class_queries_logits` × `masks_queries_logits.sigmoid()`) dikonversi menjadi peta log-probabilitas dense per piksel via batch matrix multiplication, agar kompatibel langsung dengan `NLLLoss`

### Hyperparameter Training

| Parameter | Nilai | Justifikasi |
|---|---|---|
| `lr_backbone` | 1e-5 | Backbone pretrained → fine-tuning konservatif |
| `lr_decoder` | 2e-5 | 2× backbone, bridge antara backbone dan head |
| `lr_head` | 1e-4 | Head baru → LR lebih besar untuk belajar cepat |
| `weight_decay` | 0.05 | Decoupled weight decay standar AdamW transformer |
| `batch_size` | 2 (efektif 8) | `accumulation_steps=4` untuk hemat VRAM GPU T4 |
| `grad_clip` | 1.0 | Cegah gradient explosion pada attention layer |
| `lovasz_weight` | 0.75 (warmup 3 epoch) | 75% Lovász + 25% NLL setelah fase warmup |
| `poly_power` | 0.9 | LR decay polinomial standar DeepLab/SegFormer |
| `early_stop_patience` | 10 | Hentikan jika val mIoU & smoothed loss stagnan |

---

## Hasil

Training dijalankan penuh **50 epoch** tanpa early stopping (val_mIoU stabil setelah epoch ~20).

| Epoch | Train Loss | Val Loss | Val mIoU |
|---|---|---|---|
| 1 | 1,0889 | 0,6905 | 0,3084 |
| 5 | 0,3628 | 0,5002 | 0,4616 |
| 10 | 0,3159 | 0,4792 | 0,4935 |
| 20 | 0,2923 | 0,4767 | 0,5099 |
| **28 (best)** | **0,2846** | **0,4436** | **0,5177** |
| 40 | 0,2522 | 0,4479 | 0,5138 |
| 50 | 0,2403 | 0,4355 | 0,5140 |

> **Best validation mIoU: 0,5177** pada epoch 28. Checkpoint terbaik disimpan sebagai `best_mask2former.pth`.

### Inference & Submission

Dua mode inference dijalankan dengan TTA horizontal flip pada **447 gambar test**, masing-masing menghasilkan **4.470 baris** submission (447 × 10 kelas), tervalidasi penuh tanpa baris kosong:

| File | Mode | Kelebihan |
|---|---|---|
| `submission_argmax.csv` | Argmax standar | Presisi lebih tinggi untuk kelas dominan |
| `submission_threshold.csv` | Per-class threshold | Recall lebih tinggi untuk kelas rare (vehicle, tree) |

---

## Struktur Notebook

| Cell | Deskripsi |
|---|---|
| 0 | Setup environment dan import library |
| 1 | Deteksi environment (Kaggle/Colab) dan GPU |
| 2 | Load dataset dan ekstraksi ZIP |
| 3 | Konfigurasi path dataset |
| 4 | Konfigurasi kelas, bobot, dan hyperparameter training |
| 5 | Helper decoding mask dan konversi JSON ke mask |
| 6 | Exploratory Data Analysis: distribusi kelas |
| 7 | Visualisasi sampel gambar + overlay mask |
| 8 | Analisis korelasi co-occurrence antar kelas |
| 9 | Scanning anomali label |
| 10 | Analisis ukuran objek kelas minoritas (vehicle) |
| 11 | Analisis ketajaman gambar (Laplacian variance) |
| 12 | Pipeline augmentasi train dan validasi (Albumentations) |
| 13 | BalancedBatchSampler |
| 14 | Arsitektur model: Mask2FormerFlood |
| 15 | Utilitas RLE encoding dan per-class thresholding |
| 16 | Metrik evaluasi (IoU/Dice) dan konfigurasi optimizer-scheduler |
| 17 | Fungsi inference dan generate submission (versi awal) |
| 18 | Fungsi inference dengan image processor (revisi) |
| 19 | Definisi ulang Mask2FormerFlood untuk konfigurasi training final |
| 20 | FloodSegDataset dengan kategorisasi rare/water/normal |
| 21 | EMA (Exponential Moving Average) + training loop |
| 22 | FloodSegLoss: Lovász-Softmax + NLL hybrid |
| 23 | **Training final: Mask2Former-Swin-Tiny (50 epoch)** |
| 24 | Inference Test-Time Augmentation dan pembuatan submission |
| 25 | Download link submission |

---

## Cara Menjalankan

Notebook didesain untuk berjalan di **Kaggle Notebook** dengan akselerator GPU (Tesla T4 atau lebih baik).

1. **Siapkan dataset** pastikan dataset `flood-segmentation-nexus-final` ter-attach sebagai Kaggle Dataset, atau tersedia sebagai file ZIP yang dapat diekstrak otomatis oleh Cell 2.
2. **Jalankan cell secara berurutan** dari Cell 0 hingga Cell 23 untuk melakukan training penuh dari awal.
   - Notebook mendukung **resume otomatis**: jika `last_checkpoint_baseline.pth` ditemukan di `save_dir`, training akan dilanjutkan dari epoch terakhir tanpa mengulang dari awal.
3. **Jalankan Cell 24** untuk menghasilkan prediksi pada test set (mode argmax dan per-class threshold) sekaligus file submission `.csv`.
4. **Jalankan Cell 25** untuk membuat link download submission langsung dari output notebook.

> Jika hanya ingin melakukan inference tanpa training ulang, cukup jalankan Cell 0-4 (setup & konfigurasi), Cell 14-16 (definisi model & metrik), lalu load checkpoint `best_mask2former.pth` sebelum menjalankan Cell 24.

---

## Requirements

```
torch
transformers
albumentations
opencv-python
numpy
pandas
Pillow
tqdm
wandb        # opsional, untuk experiment tracking
```

Model pretrained `facebook/mask2former-swin-tiny-ade-semantic` diunduh otomatis dari HuggingFace Hub saat pertama kali dijalankan (membutuhkan koneksi internet aktif di environment Kaggle).

---

## Struktur Output

```
/kaggle/working/
├── checkpoints_mask2former-swin-tiny-ade-semantic/
│   ├── best_baseline.pth  # checkpoint val_mIoU terbaik
│   ├── last_checkpoint_baseline.pth   # checkpoint terakhir (untuk resume)
│   └── best_mask2former.pth # kopi final dari best_baseline.pth
├── submission_argmax.csv  # 4.470 baris, mode argmax
└── submission_threshold.csv  # 4.470 baris, mode per-class threshold
```

---

## Catatan Teknis

- **Reproducibility**: seed 42 ditetapkan secara menyeluruh (`random`, `numpy`, `torch`, `cudnn.deterministic=True`), dengan trade-off sedikit penurunan kecepatan training.
- **Validasi dibagi dua**: 50% untuk early stopping selama training (`val_ids_earlystop`), 50% disisihkan untuk hyperparameter tuning (`val_ids_tuning`) agar tidak ada overfitting terhadap metrik validasi.
- **Dual early stopping signal**: training berhenti hanya jika **val_mIoU** dan **smoothed validation loss** keduanya stagnan selama `patience` epoch mencegah penghentian dini saat model masih dalam fase konsolidasi.
- **Dua versi fungsi inference** (Cell 17 dan Cell 18) dipertahankan di notebook untuk transparansi proses eksperimen; versi yang menghasilkan mIoU validasi lebih tinggi dipakai untuk submission final.
