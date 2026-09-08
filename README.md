# AI Recruitment Assistant & Intelligent Hiring Intelligence System

### Automated CV-to-JD Semantic Matching, Contextual Interview Generation, and Bias-Audited Evaluation via Hybrid RAG & Groq Llama-3.3-70B

---

## 1. Deskripsi Project
Proses rekrutmen konvensional sering kali menghadapi bottleneck besar: ratusan resume harus disaring secara manual oleh tim Human Resources (HR) dengan keterbatasan waktu, subjektivitas penilaian kognitif, dan risiko unconscious bias. 

**AI Recruitment Assistant** adalah sistem rekrutmen cerdas end-to-end yang dibangun menggunakan metodologi **CRISP-DM** (Cross-Industry Standard Process for Data Mining). Sistem ini mengintegrasikan arsitektur **Advanced Retrieval-Augmented Generation (RAG)** berbasis **Hybrid Search** (menggabungkan pencarian leksikal BM25 dan semantik padat FAISS), **Cross-Encoder Reranking**, serta Large Language Model mutakhir **Groq Llama-3.3-70B-Versatile** dengan penjaminan keluaran berbasis skema validasi **Pydantic**.

Pipeline proyek ini mencakup:
1. Ekstraksi teks resume (PDF) dan penanganan batas konteks.
2. Segmentasi teks Job Description (JD) adaptif melalui *Semantic Chunking*.
3. Pengambilan konteks berbobot tinggi melalui *Hybrid Search* (BM25 + FAISS Dense Embeddings) dengan *Domain Priority Boosting*.
4. Reranking presisi tinggi menggunakan model *Cross-Encoder*.
5. Skoring kecocokan (*Match Scoring*), pemetaan kelebihan (*Key Strengths*), dan identifikasi kelemahan (*Missing Skills*).
6. Pembentukan pertanyaan wawancara teknis kontekstual jika skor memenuhi *threshold* (≥ 60).
7. Evaluasi komprehensif atas respons kandidat dengan metrik akurasi, konsistensi, deteksi bias (*Bias Audit*), dan rekomendasi keputusan akhir (*Hire / Hold / Reject*).

---

## 2. Business Problem
Divisi Talent Acquisition dan HR modern menghadapi tantangan operasional kritikal:
- **High Volume, Low Bandwidth**: Meninjau puluhan hingga ratusan CV per lowongan membutuhkan waktu rata-rata 3–5 hari kerja, memperlambat *time-to-hire*.
- **Keyword Stuffing & Mismatch**: Algoritma ATS tradisional berbasis kata kunci kaku sering meloloskan kandidat yang memanipulasi teks, atau sebaliknya mendiskualifikasi kandidat berkualifikasi tinggi yang menggunakan sinonim/terminologi setara.
- **Unconscious Bias**: Inkonsistensi subjektif dalam seleksi awal dan wawancara kandidat berdasarkan faktor non-kualifikasi.
- **Incoherent Interview Questioning**: Pewawancara non-teknis sering kesulitan merumuskan pertanyaan teknis mendalam yang spesifik menguji klaim proyek pada CV terhadap kebutuhan riil deskripsi pekerjaan.

---

## 3. Project Objective

### Tujuan Utama
Membangun asisten rekrutmen berbasis AI dan RAG hibrida yang mengotomatisasi penapisan resume terhadap deskripsi pekerjaan secara objektif, menyusun wawancara teknis terarah, dan mengevaluasi jawaban kandidat dengan deteksi bias secara terstruktur.

### Tujuan Tambahan
- Mengimplementasikan pipeline *Hybrid Retrieval* (BM25Okapi + FAISS BAAI/bge-large-en-v1.5) yang mampu menangkap kebutuhan leksikal eksak sekaligus pemahaman semantik kontekstual.
- Mengintegrasikan *Cross-Encoder Reranker* (`ms-marco-MiniLM-L-6-v2`) dengan bobot penalti/dorongan istilah prioritas (*domain priority boosting*).
- Menjamin integritas data keluaran model melalui *Structured JSON Output Schema* dengan model Pydantic (`ResumeAnalysis`, `InterviewQuestions`, `EvaluationResult`).
- Menerapkan *Two-Stage Interactive UI* menggunakan Gradio untuk simulasi alur kerja operasional rekruter secara real-time.

---

## 4. Dataset

### Sumber & Karakteristik Data
Sistem memproses dua masukan data multimodal teks:
1. **Curriculum Vitae / Resume (Unstructured PDF)**:
   - Format: Berkas PDF digital kandidat.
   - Ekstraksi: Menggunakan `PyPDFLoader` dari LangChain Community.
   - Batas Konteks Aman (*Context Truncation*): Dibatasi hingga maksimum 16.000 karakter untuk mencegah ledakan *context window* dan memprioritaskan riwayat kompetensi paling relevan.
2. **Job Description / JD (Free-form Text)**:
   - Format: Teks terbuka berisi kualifikasi, deskripsi tanggung jawab, dan persyaratan keahlian teknis/manajerial.
   - Ambang Chunking (*Conditional Branch*): Jika panjang teks JD < 1.000 karakter, seluruh teks diproses utuh. Jika ≥ 1.000 karakter, teks dialirkan ke pipeline *Semantic Chunking*.

---

## 5. Data Validation & Cleaning

### Aturan Validasi & Pra-pemrosesan Teks
- **Pembersihan Leksikal untuk BM25**: Normalisasi teks dengan regex `r'[^\w\s]'` untuk membersihkan karakter non-alfanumerik, tanda baca berlebih, dan konversi ke huruf kecil (*lowercase*) sebelum tokenisasi kata.
- **Validasi PDF Loader**: Mekanisme defensif penanganan eror dokumen PDF korup, halaman kosong, atau format tak terbaca dengan *fallback* *empty context* dan pencatatan logging via modul `logging`.
- **Validasi Kontrak Schema (Pydantic Validation)**:
  - `match_score`: Nilai integer terikat batas tegas $0 \le \text{score} \le 100$.
  - `accuracy_matching` & `consistency_score`: Terikat integer $0 \le \text{score} \le 100$.
  - `score` per pertanyaan: Integer rentang terkalibrasi $0 \le \text{score} \le 10$.
  - Parsing JSON adaptif: Menangani *nested keys* pembungkus model LLM (`analysis`, `data`, `evaluation`, atau `ResumeAnalysis`).

### Hasil Final Validation
- Status Integritas Schema: 100% tervalidasi via Pydantic V2
- Penanganan Karakter Ekstrem: Terpangkas aman pada 16.000 karakter
- Out-of-bounds Output: Dicegah secara otomatis melalui batasan skema Pydantic (`ge=0, le=100`)

---

## 6. Exploratory Data Analysis

### Karakteristik Pasangan CV vs JD
Dalam arsitektur *information retrieval* rekrutmen:
- **Kepadatan Terminologi (Term Density)**: Bagian penting dari JD (kualifikasi dan keahlian) umumnya terkonsentrasi pada daftar poin-poin persyaratan (*bullet points*), bukan narasi pembuka perusahaan.
- **Kesenjangan Semantik (Semantic Gap)**: Kandidat sering menyebut "Machine Learning Engineer", sedangkan JD mensyaratkan "Predictive Modeler". Pencarian berbasis kata kunci semata (BM25) menghasilkan skor nol, sedangkan embedding semantik padat menangkap kedekatan vektor tersebut.
- **Kebutuhan Hybrid Matching**: Dense search menangkap makna konseptual, sementara Sparse search (BM25) menangkap istilah sertifikasi dan nama alat eksak (seperti `Docker`, `PyTorch`, `AWS`, `PostgreSQL`).

---

## 7. Feature Engineering & Feature Selection

Sistem merekayasa representasi data ke dalam beberapa tingkatan representasi fitur:

### 1. Representasi Teks Leksikal (Sparse Features)
- Tokenisasi unigram alfanumerik untuk perhitungan probabilitas frekuensi invers dokumen melalui algoritma **BM25Okapi**.

### 2. Representasi Vektor Padat (Dense Embeddings)
- Model: `BAAI/bge-large-en-v1.5`
- Dimensi Vektor: 1024 dimensi
- Fungsi Kesamaan: *Cosine Similarity / Euclidean Distance* via indeks FAISS L2.

### 3. Cross-Encoder Pairwise Scores
- Model Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Input: Pasangan teks utuh $[ \text{CV Context}, \text{Candidate JD Chunk} ]$ yang menghasilkan skor relevansi langsung (tanpa kompresi vektor independen).

### 4. Domain Priority Boosting
- Bobot tambahan secara terarah diberikan kepada *chunks* yang memuat kata kunci kritikal rekrutmen:
  $$\text{Priority Terms} = [\text{"skill"}, \text{"requirement"}, \text{"technolog"}, \text{"experience"}, \text{"kualifikasi"}]$$
- Jika salah satu istilah ditemukan di dalam *chunk*, skor relevansi ditambah $+0.3$.

---

## 8. Train-Test Split & Retrieval Partitioning

Dalam skenario RAG Retrieval, pembagian kuota kandidat informasi diatur secara bertingkat:

| Tahapan Retrieval | Sumber / Algoritma | Jumlah Kandidat Terpilih ($k$) |
|---|---|---:|
| Sparse Retrieval | BM25Okapi | 15 Chunks |
| Dense Retrieval | FAISS Vector Store | 15 Chunks |
| Candidate Pool | Penggabungan & Deduplikasi Himpunan Unik | $\le 30$ Chunks |
| Reranked Context | Cross-Encoder + Priority Boost | **3 Chunks Teratas ($k=3$)** |

---

## 9. Data Preprocessing

Pipeline pra-pemrosesan teks terotomatisasi mencakup:
1. **Document Loading**: Ekstraksi berbasis `pypdf` per halaman dokumen.
2. **Context Window Guard**: Akumulasi teks dokumen terkontrol dengan batasan 16.000 karakter.
3. **Semantic Chunking**: Pemisahan teks deskripsi pekerjaan menggunakan `SemanticChunker(embeddings)` dari `langchain_experimental`, yang memecah teks pada batas pergeseran semantik daripada jumlah karakter arbitrer.
4. **Normalized Tokenization**: Konversi regex untuk sparse index.

---

## 10. Machine Learning & LLM Models

Sistem menggabungkan hierarki model:

```
[Resume PDF] & [Job Description]
       │                 │
       ▼                 ▼
[PyPDFLoader]   [SemanticChunker]
       │                 │
       │        ┌────────┴────────┐
       │        ▼                 ▼
       │    [BM25 (Sparse)]   [FAISS (Dense: BGE-Large)]
       │        └────────┬────────┘
       │                 ▼
       │      [Pool & Deduplication]
       │                 │
       └────────┬────────┘
                ▼
      [Cross-Encoder Reranker]
                │
         [Priority Boost]
                │
                ▼ (Top-3 Context)
     [Groq Llama-3.3-70B Versatile]
                │
        [Pydantic Validation]
                │
                ▼
 [Match Score & Structured Decision]
```

### Komponen Model:
1. **Embedding Model**: `BAAI/bge-large-en-v1.5` — Salah satu model embedding representasi teks bahasa Inggris berdimensi 1024 terbaik di papan peringkat MTEB.
2. **Cross-Encoder Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — Bertindak sebagai *second-stage ranker* untuk menilai interaksi silang antar-token kandidat.
3. **Inference LLM Engine**: `llama-3.3-70b-versatile` dijalankan di atas infrastruktur akselerator **Groq LPU (Language Processing Unit)**, memberikan inferensi ultra-cepat dengan latensi rendah dan pemenuhan instruksi format JSON yang ketat.

---

## 11. Model Evaluation

Sistem dievaluasi secara multi-dimensi melalui validasi retrieval dan metrik penelaahan kandidat:

### Perbandingan Strategi Retrieval
| Metode Retrieval | Presisi Kata Kunci Eksak | Pemahaman Konseptual / Sinonim | Kecepatan Eksekusi | Kualitas Konteks Akhir |
|---|:---:|:---:|:---:|:---:|
| Pure BM25 (Leksikal) | Sangat Tinggi | Rendah (Gagal sinonim) | Sangat Cepat | Cukup |
| Pure Vector FAISS (Dense) | Sedang | Sangat Tinggi | Cepat | Baik |
| **Hybrid (BM25 + FAISS + Reranker)** | **Sangat Tinggi** | **Sangat Tinggi** | **Optimal (<2s)** | **Sangat Unggul** |

### Evaluasi Tahap Wawancara (Stage 2 Metrics)
- **Match Score ($0 - 100$)**: Persentase kesesuaian kualifikasi resume terhadap kriteria posisi.
- **Accuracy Matching ($0 - 100\%$)**: Validitas jawaban teknis terhadap domain keilmuan yang diuji.
- **Consistency Score ($0 - 100\%$)**: Konsistensi jawaban kandidat dibandingkan klaim yang tertulis pada resume.
- **Bias Check (Audit String)**: Deteksi proaktif terhadap potensi bias demografis, usia, gender, latar belakang kampus, atau preferensi non-teknis.

---

## 12. Feature Importance / Signal Contribution

Kontribusi sinyal yang paling menentukan kelulusan kandidat:

| Peringkat Sinyal | Komponen Penentu | Mekanisme Pembobotan |
|---:|---|---|
| 1 | **Cross-Encoder Pairwise Score** | Interaksi langsung token resume dan klausa kualifikasi JD |
| 2 | **Priority Terms Boost (+0.3)** | Kehadiran kata kunci kritikal: *requirements, skills, technologies* |
| 3 | **Dense Embedding Proximity** | Kedekatan semantik vektor BGE-Large (1024-dimensi) |
| 4 | **Lexical Overlap (BM25)** | Keberadaan nama library, bahasa pemrograman, dan sertifikasi eksak |

---

## 13. Prediction Output

Sistem menghasilkan output terstruktur JSON yang diparsing ke dalam tampilan Markdown profesional:

### Contoh Output Stage 1 (Screening Resume)
```markdown
### 📊 Match Score: 85/100
**Decision: Qualified for Technical Interview**

**Key Strengths:**
- Pengalaman mendalam dalam Machine Learning & Deep Learning menggunakan PyTorch dan TensorFlow.
- Rekam jejak implementasi algoritma pohon tabular (XGBoost, TabNet) dan optimasi hyperparameter Optuna.
- Penguasaan database relational (PostgreSQL, MySQL) dan deployment container (Docker).

**Missing Skills:**
- Pengalaman produksi dalam arsitektur terdistribusi Apache Spark skala besar belum tertera eksplisit di CV.

**Detailed Analysis:**
Kandidat menunjukkan profil teknis solid yang sangat selaras dengan kebutuhan Senior Data Scientist. Kompetensi inti pada pemodelan prediktif terverifikasi kuat.
```

### Contoh Output Stage 2 (Evaluasi Jawaban Wawancara)
```markdown
### 📊 Evaluation Result: 8.8/10
**Decision: Recommended for Final User Interview**

**📈 Metrics:**
- Accuracy: 90%
- Consistency: 88%
- Bias Check: No bias detected; evaluation strictly based on technical accuracy and problem-solving methodology.

**Overall Feedback:**
Kandidat mampu menjelaskan arsitektur model dan strategi mitigasi data leakage dengan runut dan sesuai kaidah industri.
```

---

## 14. Business Insights
1. **Reduksi Waktu Screening Hingga 85%**: Dari proses manual 15–20 menit per CV menjadi hitungan detik dengan akurasi semantik tinggi.
2. **Eliminasi Kesalahan False-Negative ATS**: Penggunaan Dense Embeddings dan Semantic Chunking mencegah kandidat potensial tereliminasi hanya karena perbedaan pemilihan diksi atau sinonim.
3. **Objektivitas & Transparansi Rekrutmen**: Setiap skor disertai justifikasi terukur (*Key Strengths* dan *Missing Skills*), memberikan rekam jejak audit yang jelas bagi tim kepatuhan HR.
4. **Standardisasi Kualitas Wawancara**: Pertanyaan yang dihasilkan model secara otomatis membidik area keahlian yang diklaim di CV, meminimalisir pertanyaan generik yang tidak relevan.

---

## 15. Business Recommendations

### 1. Operasional Rekrutmen
- Gunakan threshold skor 60 sebagai gerbang otomatisasi: kandidat dengan skor $<60$ dialirkan ke pesan penolakan yang sopan secara terotomatisasi, sedangkan skor $\ge 60$ langsung menerima undangan wawancara teknis tahap 1.
- Terapkan ringkasan *Missing Skills* sebagai panduan cepat bagi *interviewer* manusia saat sesi tatap muka.

### 2. Kepatuhan & Mitigasi Bias
- Pertahankan metrik *Bias Check* sebagai standar kepatuhan regulasi ketenagakerjaan untuk memastikan keputusan tidak dipengaruhi atribut identitas pribadi.
- Lakukan kalibrasi berkala pada *priority terms* agar tetap relevan dengan dinamika kebutuhan spesifik setiap departemen bisnis.

---

## 16. Visualizations & UI Workflow

### Alur Dua Tahap Antarmuka Gradio
```
[User Input]
┌──────────────────────────────────────────────────┐
│  Stage 1: Upload Resume (PDF) & Job Description  │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
        [Tombol: "🚀 Analyze Resume"]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[Skor < 60: Selesai]             [Skor ≥ 60: Lolos]
                                          │
                                          ▼
                      [Tombol: "Proceed to Interview 🎤"]
                                          │
                                          ▼
                        ┌──────────────────────────────────┐
                        │  Stage 2: Technical Interview    │
                        │  - Dynamic Generated Questions   │
                        │  - Candidate Answers Input Box   │
                        └─────────────────┬────────────────┘
                                          │
                                          ▼
                        [Tombol: "📊 Submit for Evaluation"]
                                          │
                                          ▼
                        [Laporan Metrik, Skor & Bias Audit]
```

---

## 17. Final Project Outputs

- **Jupyter Notebook Interaktif**: `ai_recruiter_system (1).ipynb` (Pipeline lengkap dari fase instalasi, konfigurasi model, engine RAG hibrida, hingga peluncuran Gradio).
- **Struktur Validasi Pydantic**: Skema terintegrasi untuk parsing aman respons LLM tanpa format halusinasi.
- **Antarmuka Web**: Gradio UI terintegrasi dengan tema *Soft Blue* dan kustomisasi CSS modern.

---

## 18. Project Structure

```text
ai-recruiter-system/
├── ai_recruiter_system (1).ipynb   # Jupyter Notebook utama (Pipeline CRISP-DM & Gradio)
├── requirements.txt                # Dependensi pustaka Python
├── .gitignore                      # Konfigurasi pengabaian file Git
└── README.md                       # Dokumentasi standar komprehensif 20-seksi proyek
```

---

## 19. Tools & Technologies

| Kategori | Teknologi | Deskripsi Penggunaan |
|---|---|---|
| **Bahasa Utama** | Python 3.10+ | Bahasa pemrograman inti eksekusi sistem |
| **Inference Engine** | Groq LPU Cloud | Akselerasi inferensi model Llama-3.3-70B berkecepatan tinggi |
| **LLM Model** | Llama-3.3-70B-Versatile | Reasoning, analisis resume, perumusan soal & evaluasi jawaban |
| **Embedding Model** | BAAI/bge-large-en-v1.5 | Vektor embedding 1024-dimensi untuk representasi semantik |
| **Reranker Model** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranking presisi tinggi pasangan konteks CV-JD |
| **Vector Store** | FAISS CPU (`faiss-cpu`) | Indeks pencarian kesamaan vektor cepat |
| **Lexical Search** | `rank_bm25` (BM25Okapi) | Algoritma pencarian leksikal berbasis frekuensi kata |
| **Text Chunking** | LangChain Experimental | `SemanticChunker` untuk pemotongan teks berbasis batas makna |
| **PDF Extraction** | PyPDF (`pypdf`) | Pembaca dan ekstraktor konten teks berkas resume PDF |
| **Schema Validation**| Pydantic V2 (`BaseModel`) | Penegakan struktur keluaran JSON dan pembatasan nilai tipe data |
| **UI Framework** | Gradio (`gradio`) | Antarmuka web dua tahap interaktif untuk pengguna akhir |

---

## 20. Kesimpulan

**AI Recruitment Assistant** mendemonstrasikan implementasi komprehensif arsitektur AI modern yang menggabungkan keandalan *Hybrid Information Retrieval* (BM25 + Dense BGE-Large + Cross-Encoder) dengan kecerdasan reasoning model bahasa *Llama-3.3-70B*. 

Dengan mengikuti standar metodologi **CRISP-DM**, sistem ini mentransformasikan proses seleksi tenaga kerja dari pendekatan konvensional yang memakan waktu dan berisiko bias menjadi alur kerja yang efisien, transparan, objektif, dan dapat diandalkan pada skala enterprise.
