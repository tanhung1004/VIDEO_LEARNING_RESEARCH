# VIDEO_LEARNING_RESEARCH — Module 1: Learning Concept Detection trong Video Bài Giảng

## 1. Giới thiệu dự án

### 1.1. Bài toán và câu hỏi nghiên cứu

Module 1 xây dựng hệ thống tự động phát hiện **learning concept** (khái niệm
lập trình — ví dụ `SELECT`, `LEFT JOIN`, `for loop`, `pointer`...) xuất hiện
trong video bài giảng lập trình thuộc 4 subject: **SQL, Python, Java, C++**.

Câu hỏi nghiên cứu chính:

> Việc bổ sung bằng chứng từ **OCR** (nội dung chữ hiển thị trên màn hình —
> code, slide) có cải thiện khả năng phát hiện learning concept so với chỉ
> dùng **transcript** (lời giảng) hay không, và cải thiện ở mức độ nào có thể
> chứng minh được bằng thống kê?

Cách tiếp cận: với mỗi video, transcript (và/hoặc OCR) được chia thành các
đoạn (chunk) theo thời gian, mỗi chunk được biểu diễn bằng **LDA** (topic
modeling) và **LSA** (semantic similarity) so với danh mục concept
(`concept_catalog.csv`), hai điểm số được **fuse** lại thành một điểm cuối,
gộp lên cấp video bằng **aggregation**, rồi **threshold** để ra dự đoán
concept nào xuất hiện trong video nào. Kết quả được so với **ground truth**
đã gán nhãn thủ công bằng Precision / Recall / F1 và các chỉ số liên quan
(MCC, PR-AUC, ROC-AUC, mAP...).

### 1.2. Dự án đã đi qua những giai đoạn nào

**Giai đoạn 1 — Pilot (baseline vs proposed, 5 video).**
Bản chứng minh khái niệm đầu tiên: 5 video, chunk 60 giây, fusion
`0.4·LDA + 0.6·LSA`, threshold `top2_mean_score ≥ 0.40`. So sánh
**baseline** (transcript-only) với **proposed** (transcript + OCR, OCR lấy
từ keyframe cào bằng Selenium, không tải video). Code nằm ở
`module1/baseline/` và `module1/proposed/`.

**Giai đoạn 2 — Mở rộng dataset.**
Mở rộng lên **40 video DEV**, cân bằng theo 4 subject, gắn với 39 concept và
153 ground-truth pair. Xây 5-fold video-level cross-validation (không để
chunk của cùng 1 video lọt sang 2 fold khác nhau).

**Giai đoạn 3 — Ablation study A–F cho phần OCR.**
Để tách bạch OCR đóng góp gì, ai đóng góp (full-frame hay ROI), cần cleaning
hay không, dự án thiết kế 6 biến thể:

| Variant | Mô tả |
|---|---|
| **A** | Transcript-only |
| **B** | OCR-only, full-frame, đã clean |
| **C** | Transcript + OCR full-frame **raw** (chưa clean) |
| **D** | Transcript + OCR full-frame đã clean |
| **E** | Transcript + OCR ROI (vùng nội dung chính, không lấy UI/sidebar) đã clean |
| **F** | Late fusion `mean(D, E)` |

Code nằm ở `module1/ocr_best_integration/`.

**Giai đoạn 4 — P1: validation nghiêm ngặt và khoá config.**
Dùng nested cross-validation (32 train / 8 held-out mỗi outer fold, fit chỉ
trên train, transform trên held-out), kiểm tra độ nhạy theo chunk size,
kiểm định thống kê (bootstrap CI, Wilcoxon, paired t-test, McNemar) để chọn
và **khoá (lock)** cấu hình cuối cùng — gọi là **NEW protocol**:

```
chunk = 180 giây
LDA weight = 0, LSA weight = 1   (LSA-only)
aggregation = max
threshold = 0.50
```

so với **OLD protocol** (cấu hình pilot ban đầu: 60s / 0.4 LDA + 0.6 LSA).
DEV macro video F1 của transcript-only dưới NEW protocol: **≈ 0.728**.
Code nằm ở `module1/p0_dataset/`, `module1/p1_validation/`,
`module1/scaffold40_compare/`.

**Giai đoạn 5 — Re-evaluate OCR trên NEW protocol (đang làm).**
D, E, F đã được build lại và đánh giá dưới NEW protocol
(`module1/ocr_best_integration/09`–`13`): NEW combined F đạt Macro video F1
**0.668** so với OLD combined **0.573** (Δ +0.095, bootstrap 95% CI
`[0.039, 0.154]`, Wilcoxon p = 0.0017) — cải thiện chủ yếu nhờ giảm false
positive chứ không phải tăng recall.

B và C (2 variant còn thiếu dưới NEW protocol) đang được bổ sung bằng
`14_build_variant_b_ocr_only_new.py`, `15_build_variant_c_transcript_plus_raw_ocr_new.py`
và đánh giá gộp cả 6 variant bằng
`16_evaluate_variant_a_to_f_new.py` — để có **bảng so sánh A–F đầy đủ, cùng
một protocol, cùng aggregation/threshold**, thay vì chỉ so được D/E/F như
trước.

### 1.3. Hướng tiếp theo (theo báo cáo tiến độ gần nhất)

- Freeze/audit toàn bộ cấu hình P2/OCR để không còn chỉnh trên tập DEV.
- Sau khi freeze, mở rộng đánh giá ra ngoài tập DEV theo một protocol tách
  biệt (không dùng lại dữ liệu đã freeze để tinh chỉnh threshold/chunk/OCR).
- Tiếp tục khảo sát literature về dense semantic representation,
  OCR-noise-robust text modeling, selective visual evidence/keyframe
  weighting, cross-modal video-text alignment — cân nhắc cho hướng mở rộng
  tiếp theo (không phải thay ngay hệ hiện tại).
- Cân nhắc bổ sung public benchmark / external dataset như một nghiên cứu mở
  rộng, tách khỏi pipeline đã freeze.
- Hoàn thiện ablation nhân quả cho OCR: Transcript-only (A) vs +D vs +E vs F
  trên cùng một frozen transcript config — đây chính là việc bảng A–F ở
  bước 16 đang phục vụ.

### 1.4. Reproducibility

Repository: `github.com/tanhung1004/VIDEO_LEARNING_RESEARCH`
(branch `ocr-best-integration` đã merge vào `main` qua PR #1).
Tag đáng chú ý: `p1-scientific-validation-v1`, `p1-final-config-v1`,
`ocr-d-e-f-scores-v1`, `ocr-d-vs-f-comparison-v1`, `old-vs-new-combined-dev-v1`.

---

## 2. Cấu trúc project

```text
VIDEO_LEARNING_RESEARCH/
│
├── data/
│   ├── concepts/
│   │   └── concept_catalog.csv
│   │
│   ├── ground_truth/
│   │   └── ground_truth_concepts.csv
│   │
│   ├── raw/
│   │   ├── videos.csv
│   │   ├── video_candidates.csv
│   │   ├── video_direct_links.csv
│   │   └── transcripts/
│   │
│   ├── splits/
│   │   └── dev_cv_folds.csv
│   │
│   └── processed/
│       ├── transcript/
│       │   ├── transcript_documents.csv
│       │   ├── transcript_preprocessed.csv
│       │   └── transcript_chunks.csv
│       │
│       ├── keyframes/                     # keyframe theo từng video (v1, v2, ...)
│       │
│       ├── ocr/
│       │   └── ocr_cleaned.csv            # OCR pilot (giai đoạn 1, quy mô nhỏ)
│       │
│       ├── ocr_best_integration/
│       │   ├── ocr_fullframe_cleaned.csv  # OCR full-frame đã clean (variant D/B)
│       │   ├── ocr_fullframe_raw.csv      # OCR full-frame CHƯA clean (variant C)
│       │   └── ocr_roi_cleaned.csv        # OCR vùng ROI đã clean (variant E)
│       │
│       └── transcript_ocr/
│           └── transcript_ocr_chunks.csv
│
├── module1/
│   ├── baseline/               # Giai đoạn 1 — transcript-only (pilot 5 video)
│   ├── proposed/                # Giai đoạn 1 — transcript + OCR (pilot 5 video)
│   ├── p0_dataset/              # Giai đoạn 2 — mở rộng 40 video DEV + ground truth
│   ├── p1_validation/           # Giai đoạn 4 — nested CV, statistical validation, khoá NEW protocol
│   ├── scaffold40_compare/      # So sánh transcript-only OLD (60s) vs NEW (180s)
│   ├── ocr_best_integration/    # Giai đoạn 3 & 5 — ablation A-F, re-evaluate NEW protocol
│   │   ├── 00 … 13_...py        # build D/E/F, so sánh OLD vs NEW combined
│   │   ├── 14_build_variant_b_ocr_only_new.py
│   │   ├── 15_build_variant_c_transcript_plus_raw_ocr_new.py
│   │   └── 16_evaluate_variant_a_to_f_new.py
│   │
│   ├── results/                 # Output của tất cả các giai đoạn trên (đã có sẵn phần lớn)
│   │   ├── baseline/  proposed/  comparison/
│   │   ├── p0_dataset/  p1_validation/  scaffold40_compare/
│   │   └── ocr_best_integration/
│   │
│   └── src/
│       ├── preprocessing.py
│       ├── lda_model.py
│       └── lsa_model.py
│
└── README.md
```

---

## 3. Hướng dẫn chạy

> Phần này mô tả cách chạy **pipeline pilot ban đầu** (baseline vs proposed,
> 5 video, giai đoạn 1). Đây là bản dễ chạy nhất để hiểu luồng xử lý cơ bản
> của dự án. Muốn tái hiện kết quả **hiện tại** (40 video, ablation A-F, NEW
> protocol) thì chạy theo đúng thứ tự 5 giai đoạn ở mục 1.2, chi tiết từng
> file trong mỗi giai đoạn nằm ngay trong `module1/<giai_đoạn>/`.

> **Quan trọng:** Không chạy `module1/proposed/01_download_videos.py`.
> Pipeline hiện tại **không tải full video**. Keyframe được lấy trực tiếp từ
> YouTube bằng Selenium + Chrome headless.

### 3.1. Chuẩn bị môi trường

Mở PowerShell tại thư mục gốc project:

```powershell
cd path\to\VIDEO_LEARNING_RESEARCH
```

Cài các thư viện Python cần dùng:

```powershell
pip install pandas numpy scikit-learn selenium pillow pytesseract matplotlib youtube-transcript-api
```

Kiểm tra Python:

```powershell
python --version
```

Pipeline keyframe cần **Google Chrome**.

Pipeline OCR cần **Tesseract OCR**. Kiểm tra:

```powershell
Test-Path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Nếu trả về `True` thì dùng được. `03_ocr_keyframes.py` đã có logic tự tìm
`tesseract.exe`.

### 3.2. File đầu vào cần có trước khi chạy

**`data/raw/videos.csv`** — tối thiểu các cột `video_id,subject,title,url,status`.
Video cần xử lý phải có `status = selected`.

```csv
video_id,subject,title,url,status
v1,SQL,...,https://www.youtube.com/watch?v=...,selected
v2,Python,...,https://www.youtube.com/watch?v=...,selected
```

**`data/concepts/concept_catalog.csv`** — cột `subject,concept,description`.

**`data/ground_truth/ground_truth_concepts.csv`** — tối thiểu cột
`video_id,subject,concept`. Ground truth chỉ dùng ở bước evaluation, không
dùng để train LDA/LSA hay chỉnh OCR.

### 3.3. PHẦN A — Baseline: Transcript-only

Baseline phải chạy trước vì proposed dùng lại `transcript_chunks.csv`, và
Step 12 cần kết quả baseline để so sánh.

| Step | Lệnh | Mục đích | Output chính |
|---|---|---|---|
| 01 | `python module1/baseline/01_get_transcripts.py` | YouTube URL → transcript | `data/raw/transcripts/*.csv` |
| 02 | `python module1/baseline/02_build_documents.py` | Gom transcript thành document/video | `data/processed/transcript/transcript_documents.csv` |
| 03 | `python module1/baseline/03_preprocess_documents.py` | Lowercase / cleaning / stopword | `data/processed/transcript/transcript_preprocessed.csv` |
| 04 | `python module1/baseline/04_build_chunks.py` | Chia chunk 60s theo timestamp | `data/processed/transcript/transcript_chunks.csv` (51 chunks với dữ liệu pilot) |
| 05 | `python module1/baseline/05_run_lda.py` | CountVectorizer → LDA → document-topic distribution | `results/baseline/lda_topics.csv`, `lda_chunk_topics.csv` |
| 06 | `python module1/baseline/06_run_lsa.py` | Chunk vs concept+description → LSA similarity | `results/baseline/lsa_concept_scores.csv`, `lsa_top_matches.csv` |
| 07 | `python module1/baseline/07_run_lda_concept_matching.py` | Topic-word × concept vector → topic-concept affinity → LDA concept score | `results/baseline/lda_concept_scores.csv`, `lda_top_concepts.csv` |
| 08 | `python module1/baseline/08_fuse_lda_lsa.py` | `final_score = 0.4·lda_norm + 0.6·lsa_norm` (normalize theo video_id+chunk_id) | `results/baseline/fusion_scores.csv`, `fusion_top_concepts.csv` |
| 09 | `python module1/baseline/09_build_video_concept_evidence.py` | Gom chunk-level evidence lên video+concept (`max/mean/top2_mean/top1_ratio`...) | `results/baseline/video_concept_evidence.csv` |
| 10 | `python module1/baseline/10_predict_evaluate_baseline.py` | Threshold `top2_mean_score ≥ 0.40` → predict → so GT → Precision/Recall/F1 | `results/baseline/evaluation_metrics.csv` |

### 3.4. PHẦN B — Proposed: Transcript + OCR

| Step | Lệnh | Mục đích | Output chính |
|---|---|---|---|
| 02 | `python module1/proposed/02_extract_keyframes_from_youtube.py` | Cào keyframe mỗi 10s bằng Selenium (không tải video) | `data/processed/keyframes/` |
| 03 | `python module1/proposed/03_ocr_keyframes.py` | OCR từng keyframe bằng Tesseract | OCR raw |
| 04 | `python module1/proposed/04_clean_ocr_text.py` | Clean text OCR | `data/processed/ocr/ocr_cleaned.csv` |
| 05 | `python module1/proposed/05_build_transcript_ocr_chunks.py` | Merge transcript + OCR theo timestamp, chia chunk 60s | `data/processed/transcript_ocr/transcript_ocr_chunks.csv` |
| 06 | `python module1/proposed/06_run_lda_transcript_ocr.py` | LDA trên transcript+OCR | — |
| 07 | `python module1/proposed/07_run_lsa_transcript_ocr.py` | LSA trên transcript+OCR | — |
| 08 | `python module1/proposed/08_run_lda_concept_matching.py` | LDA concept matching (method `transcript_ocr`) | `results/proposed/lda_concept_scores.csv` |
| 09 | `python module1/proposed/09_fuse_lda_lsa.py` | Giữ đúng công thức baseline `0.4·LDA + 0.6·LSA` | `results/proposed/fusion_scores.csv` (51 chunks × top3 = 153 rows) |
| 10 | `python module1/proposed/10_build_video_concept_evidence.py` | Chunk-level → video+concept evidence | `results/proposed/video_concept_evidence.csv` |
| 11 | `python module1/proposed/11_predict_evaluate_proposed.py` | Giữ nguyên threshold `0.40` (không tune riêng cho OCR) | `results/proposed/evaluation_metrics.csv` |

### 3.5. PHẦN C — So sánh và trực quan hoá (pilot)

```powershell
python module1/proposed/12_compare_baseline_vs_ocr.py
python module1/proposed/13_visualize_comparison.py
```

Output: `results/comparison/baseline_vs_ocr.csv` và
`results/comparison/figures/{precision_recall_f1,tp_fp_fn,metrics_difference}.png`.

Nguyên tắc để so sánh công bằng: baseline và proposed phải giữ **giống
nhau** ở mọi thứ downstream (chunk duration, preprocessing, LDA config, LSA,
concept catalog, fusion weight, threshold, ground truth, evaluation
metric) — khác biệt duy nhất được kiểm thử là có OCR hay không, và **không**
tune threshold riêng cho proposed chỉ để tăng F1.

### 3.6. Chạy pipeline hiện tại (40 video, ablation A-F, NEW protocol)

Sau khi đã có `data/processed/ocr_best_integration/*.csv` (đã cào/OCR sẵn):

```powershell
# Ablation OCR-only (variant B)
python module1/ocr_best_integration/14_build_variant_b_ocr_only_new.py

# Transcript + OCR raw (variant C) - cần ocr_fullframe_raw.csv
python module1/ocr_best_integration/15_build_variant_c_transcript_plus_raw_ocr_new.py

# So sánh đầy đủ A-F trên cùng NEW protocol (180s / LSA-only / max / th=0.50)
python module1/ocr_best_integration/16_evaluate_variant_a_to_f_new.py
```

Kết quả: `results/ocr_best_integration/combined_system_comparison/variant_a_to_f_evaluation/a_to_f_overall_metrics.csv`.

### 3.7. Chỉ muốn chạy lại phần cuối (không chạy lại từ đầu)

Nếu transcript, keyframe, OCR và intermediate file đã tồn tại:

```powershell
python module1/proposed/09_fuse_lda_lsa.py
python module1/proposed/10_build_video_concept_evidence.py
python module1/proposed/11_predict_evaluate_proposed.py
python module1/proposed/12_compare_baseline_vs_ocr.py
python module1/proposed/13_visualize_comparison.py
```

Chỉ muốn vẽ lại biểu đồ:

```powershell
python module1/proposed/13_visualize_comparison.py
```

---

## 4. File kết quả quan trọng nhất

**Pilot (giai đoạn 1):**
```
module1/results/baseline/evaluation_metrics.csv
module1/results/proposed/evaluation_metrics.csv
module1/results/comparison/baseline_vs_ocr.csv
module1/results/comparison/figures/
```

**Hiện tại (giai đoạn 4-5, NEW protocol):**
```
module1/results/p1_validation/P1_FINAL_LOCK_SUMMARY.txt
module1/results/ocr_best_integration/combined_system_comparison/final_evaluation/OLD_VS_NEW_COMBINED_SUMMARY.txt
module1/results/ocr_best_integration/combined_system_comparison/variant_a_to_f_evaluation/a_to_f_overall_metrics.csv
```
