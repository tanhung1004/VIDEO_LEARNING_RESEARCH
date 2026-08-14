# VIDEO LEARNING RESEARCH — HƯỚNG DẪN CHẠY MODULE 1 TỪ A → Z

README này mô tả toàn bộ quy trình chạy **Module 1** theo đúng pipeline hiện tại:

- **Baseline:** Transcript-only
- **Proposed:** Transcript + OCR
- Cùng dùng LDA + LSA
- Cùng fusion `0.4 * LDA + 0.6 * LSA`
- Cùng prediction threshold `top2_mean_score >= 0.40`
- Cuối cùng so sánh Precision / Recall / F1 và trực quan hóa kết quả

> **Quan trọng:** Không chạy `module1/proposed/01_download_videos.py`. Pipeline hiện tại **không tải full video**. Keyframe được lấy trực tiếp từ YouTube bằng Selenium + Chrome headless.

---

## 1. Cấu trúc project chính

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
│   │   └── transcripts/
│   │       ├── v1_transcript.csv
│   │       ├── v2_transcript.csv
│   │       ├── v3_transcript.csv
│   │       ├── v4_transcript.csv
│   │       └── v5_transcript.csv
│   │
│   └── processed/
│       ├── transcript/
│       │   ├── transcript_documents.csv
│       │   ├── transcript_preprocessed.csv
│       │   └── transcript_chunks.csv
│       │
│       ├── keyframes/
│       │   ├── v1/
│       │   ├── v2/
│       │   ├── v3/
│       │   ├── v4/
│       │   └── v5/
│       │
│       ├── ocr/
│       │   └── ocr_cleaned.csv
│       │
│       └── transcript_ocr/
│           └── transcript_ocr_chunks.csv
│
├── module1/
│   ├── baseline/
│   │   ├── 01_get_transcripts.py
│   │   ├── 02_build_documents.py
│   │   ├── 03_preprocess_documents.py
│   │   ├── 04_build_chunks.py
│   │   ├── 05_run_lda.py
│   │   ├── 06_run_lsa.py
│   │   ├── 07_run_lda_concept_matching.py
│   │   ├── 08_fuse_lda_lsa.py
│   │   ├── 09_build_video_concept_evidence.py
│   │   └── 10_predict_evaluate_baseline.py
│   │
│   ├── proposed/
│   │   ├── 01_download_videos.py
│   │   ├── 02_extract_keyframes_from_youtube.py
│   │   ├── 03_ocr_keyframes.py
│   │   ├── 04_clean_ocr_text.py
│   │   ├── 05_build_transcript_ocr_chunks.py
│   │   ├── 06_run_lda_transcript_ocr.py
│   │   ├── 07_run_lsa_transcript_ocr.py
│   │   ├── 08_run_lda_concept_matching.py
│   │   ├── 09_fuse_lda_lsa.py
│   │   ├── 10_build_video_concept_evidence.py
│   │   ├── 11_predict_evaluate_proposed.py
│   │   ├── 12_compare_baseline_vs_ocr.py
│   │   └── 13_visualize_comparison.py
│   │
│   ├── results/
│   │   ├── baseline/
│   │   ├── proposed/
│   │   └── comparison/
│   │
│   └── src/
│       ├── preprocessing.py
│       ├── lda_model.py
│       └── lsa_model.py
│
└── README.md
```

---

# 2. Chuẩn bị môi trường

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

Nếu trả về:

```text
True
```

thì dùng được.

`03_ocr_keyframes.py` hiện đã có logic tự tìm `tesseract.exe`.

---

# 3. File đầu vào cần có trước khi chạy

## `data/raw/videos.csv`

Phải có ít nhất các cột:

```text
video_id,subject,title,url,status
```

Các video cần xử lý phải có:

```text
status = selected
```

Ví dụ:

```csv
video_id,subject,title,url,status
v1,SQL,...,https://www.youtube.com/watch?v=...,selected
v2,Python,...,https://www.youtube.com/watch?v=...,selected
```

## `data/concepts/concept_catalog.csv`

Chứa concept và description theo subject.

Ví dụ cấu trúc:

```text
subject,concept,description
```

## `data/ground_truth/ground_truth_concepts.csv`

Ground truth để đánh giá cuối cùng.

Tối thiểu cần:

```text
video_id,subject,concept
```

> Ground truth chỉ dùng ở bước evaluation. Không dùng để train LDA/LSA hay chỉnh OCR.

---

# PHẦN A — BASELINE: TRANSCRIPT-ONLY

Baseline phải chạy trước vì proposed sẽ dùng lại `transcript_chunks.csv`, đồng thời Step 12 cần kết quả baseline để so sánh.

---

## Step 01 — Lấy transcript

Chạy:

```powershell
python module1/baseline/01_get_transcripts.py
```

Mục đích:

```text
YouTube URL
→ transcript
→ lưu transcript theo từng video
```

Output chính:

```text
data/raw/transcripts/
├── v1_transcript.csv
├── v2_transcript.csv
├── ...
```

---

## Step 02 — Build transcript documents

Chạy:

```powershell
python module1/baseline/02_build_documents.py
```

Mục đích:

```text
raw transcript
→ gom thành document theo video
```

Output:

```text
data/processed/transcript/transcript_documents.csv
```

---

## Step 03 — Preprocess transcript

Chạy:

```powershell
python module1/baseline/03_preprocess_documents.py
```

Mục đích:

```text
raw text
→ lowercase / cleaning / stopword removal
→ processed text
```

Output:

```text
data/processed/transcript/transcript_preprocessed.csv
```

---

## Step 04 — Chia transcript thành chunk 60 giây

Chạy:

```powershell
python module1/baseline/04_build_chunks.py
```

Mục đích:

```text
Transcript
→ chia theo timestamp
→ mỗi chunk khoảng 60 giây
```

Output:

```text
data/processed/transcript/transcript_chunks.csv
```

Với bộ dữ liệu hiện tại, sanity check:

```text
51 chunks
```

---

## Step 05 — Chạy LDA baseline

Chạy:

```powershell
python module1/baseline/05_run_lda.py
```

Mục đích:

```text
processed transcript chunks
→ CountVectorizer
→ LDA topic modelling
→ document-topic distribution
```

Output trong:

```text
module1/results/baseline/
```

Ví dụ:

```text
lda_topics.csv
lda_chunk_topics.csv
```

---

## Step 06 — Chạy LSA baseline

Chạy:

```powershell
python module1/baseline/06_run_lsa.py
```

Mục đích:

```text
transcript chunk
     vs
concept + description
     ↓
LSA semantic similarity
```

Output:

```text
module1/results/baseline/
├── lsa_concept_scores.csv
└── lsa_top_matches.csv
```

---

## Step 07 — LDA concept matching

Chạy:

```powershell
python module1/baseline/07_run_lda_concept_matching.py
```

Mục đích:

```text
LDA topic-word distribution
        +
concept vector trong cùng vocabulary
        ↓
topic-concept affinity
        ↓
document-topic × topic-concept
        ↓
LDA concept score
```

Output:

```text
module1/results/baseline/
├── lda_topics_step7.csv
├── lda_topic_concept_affinity.csv
├── lda_concept_scores.csv
└── lda_top_concepts.csv
```

> Đây là LDA concept matching đã sửa. Không dùng cách cũ gây score gần `0.99` hàng loạt.

---

## Step 08 — Fuse LDA + LSA

Chạy:

```powershell
python module1/baseline/08_fuse_lda_lsa.py
```

Công thức:

```text
final_score
=
0.4 * lda_norm
+
0.6 * lsa_norm
```

Normalization được thực hiện theo từng:

```text
video_id + chunk_id
```

Output:

```text
module1/results/baseline/
├── fusion_scores.csv
└── fusion_top_concepts.csv
```

---

## Step 09 — Build video-level concept evidence

Chạy:

```powershell
python module1/baseline/09_build_video_concept_evidence.py
```

Mục đích:

```text
chunk-level evidence
→ gom lên video + concept
```

Các feature chính:

```text
max_final_score
mean_final_score
top2_mean_score
top3_count
top3_ratio
top1_count
top1_ratio
max_lsa_score
max_lda_score
best_chunk_id
best_start_sec
best_end_sec
```

Output:

```text
module1/results/baseline/video_concept_evidence.csv
```

---

## Step 10 — Predict + evaluate baseline

Chạy:

```powershell
python module1/baseline/10_predict_evaluate_baseline.py
```

Prediction rule:

```text
top2_mean_score >= 0.40
```

Output:

```text
module1/results/baseline/
├── predicted_concepts.csv
├── evaluation_details.csv
└── evaluation_metrics.csv
```

Baseline hoàn tất tại đây.

---

# PHẦN B — PROPOSED: TRANSCRIPT + OCR

> **Không chạy `module1/proposed/01_download_videos.py`.**
>
> File này là hướng thử cũ và hiện không nằm trong pipeline chính.
>
> Cũng không cần `video_direct_links.csv`.

Pipeline hiện tại:

```text
YouTube URL
→ Selenium + Chrome headless
→ keyframe mỗi 10 giây
→ OCR
→ clean OCR
→ merge OCR vào transcript chunk theo timestamp
→ LDA
→ LSA
→ fusion
→ evaluation
```

---

## Step 02 — Extract keyframes trực tiếp từ YouTube

Chạy:

```powershell
python module1/proposed/02_extract_keyframes_from_youtube.py
```

Mục đích:

```text
videos.csv
→ mở YouTube bằng Chrome headless
→ lấy duration
→ mở video tại timestamp 0s, 10s, 20s, ...
→ screenshot riêng HTML5 video element
→ lưu PNG
```

Không tải full MP4.

Output ảnh:

```text
data/processed/keyframes/
├── v1/
├── v2/
├── v3/
├── v4/
└── v5/
```

Summary:

```text
module1/results/proposed/keyframe_summary.csv
```

Với dataset hiện tại, lần chạy thành công đã thu được:

```text
291 keyframes
0 failed
```

Nếu `failed` nhiều, không chạy OCR ngay; kiểm tra Step 02 trước.

---

## Step 03 — OCR keyframes

Chạy:

```powershell
python module1/proposed/03_ocr_keyframes.py
```

Mục đích:

```text
PNG keyframe
→ grayscale
→ contrast
→ upscale
→ sharpen
→ Tesseract OCR
```

Output:

```text
module1/results/proposed/ocr_text.csv
```

Các cột chính:

```text
video_id
subject
frame_id
timestamp_sec
image_file
ocr_text
ocr_char_count
ocr_word_count
ocr_status
```

Sanity check hiện tại:

```text
Total: 291
OCR Failed: 0
```

---

## Step 04 — Clean OCR text

Chạy:

```powershell
python module1/proposed/04_clean_ocr_text.py
```

Mục đích:

```text
raw OCR
→ remove URL
→ remove known UI noise
→ normalize OCR text
→ giữ code / technical text
```

Không sử dụng:

```text
ground truth
concept matching
F1
```

nên bước này không tune theo đáp án.

Output:

```text
data/processed/ocr/ocr_cleaned.csv
```

Summary:

```text
module1/results/proposed/ocr_clean_summary.csv
```

---

## Step 05 — Merge Transcript + OCR theo timestamp

Chạy:

```powershell
python module1/proposed/05_build_transcript_ocr_chunks.py
```

Mục đích:

```text
baseline transcript_chunks.csv
        +
clean OCR
        ↓
match bằng video_id + timestamp
        ↓
mỗi OCR frame được đưa vào đúng chunk 60 giây
        ↓
remove near-duplicate OCR
        ↓
same preprocessing as baseline
```

Output:

```text
data/processed/transcript_ocr/transcript_ocr_chunks.csv
```

Summary:

```text
module1/results/proposed/transcript_ocr_combine_summary.csv
```

Sanity check:

```text
Tổng chunk proposed phải bằng baseline.
```

Với dataset hiện tại:

```text
51 chunks
```

---

## Step 06 — LDA cho Transcript + OCR

Chạy:

```powershell
python module1/proposed/06_run_lda_transcript_ocr.py
```

Mục đích:

```text
Transcript + OCR chunks
→ cùng LDA configuration như baseline
```

Output:

```text
module1/results/proposed/
├── lda_topics.csv
└── lda_chunk_topics.csv
```

Không đổi:

```text
NUM_TOPICS
CountVectorizer config
max_iter
learning_method
random_state
```

so với baseline.

---

## Step 07 — LSA cho Transcript + OCR

Chạy:

```powershell
python module1/proposed/07_run_lsa_transcript_ocr.py
```

Mục đích:

```text
Transcript+OCR chunk
      vs
concept + description
      ↓
LSA semantic similarity
```

Output:

```text
module1/results/proposed/
├── lsa_concept_scores.csv
└── lsa_top_matches.csv
```

Method:

```text
transcript_ocr
```

---

## Step 08 — LDA concept matching cho Transcript + OCR

Chạy:

```powershell
python module1/proposed/08_run_lda_concept_matching.py
```

Mục đích:

```text
Transcript+OCR
→ LDA topics
→ topic-word distribution
→ concept vector trong cùng vocabulary
→ topic-concept affinity
→ document-concept score
```

Output:

```text
module1/results/proposed/
├── lda_topics_step8.csv
├── lda_topic_concept_affinity.csv
├── lda_concept_scores.csv
└── lda_top_concepts.csv
```

---

## Step 09 — Fuse LDA + LSA cho proposed

Chạy:

```powershell
python module1/proposed/09_fuse_lda_lsa.py
```

Giữ đúng công thức baseline:

```text
final_score
=
0.4 * lda_norm
+
0.6 * lsa_norm
```

Output:

```text
module1/results/proposed/
├── fusion_scores.csv
└── fusion_top_concepts.csv
```

Sanity check hiện tại:

```text
51 chunks × Top 3 = 153 rows
```

trong:

```text
fusion_top_concepts.csv
```

---

## Step 10 — Build video-level concept evidence

Chạy:

```powershell
python module1/proposed/10_build_video_concept_evidence.py
```

Mục đích:

```text
chunk-level fusion scores
→ video + concept evidence
```

Output:

```text
module1/results/proposed/video_concept_evidence.csv
```

---

## Step 11 — Predict + evaluate Transcript + OCR

Chạy:

```powershell
python module1/proposed/11_predict_evaluate_proposed.py
```

Giữ nguyên threshold baseline:

```text
top2_mean_score >= 0.40
```

Không tune threshold riêng cho OCR.

Output:

```text
module1/results/proposed/
├── predicted_concepts.csv
├── evaluation_details.csv
└── evaluation_metrics.csv
```

---

# PHẦN C — SO SÁNH BASELINE VS OCR

## Step 12 — Tạo bảng comparison

Chạy:

```powershell
python module1/proposed/12_compare_baseline_vs_ocr.py
```

Input:

```text
module1/results/baseline/evaluation_metrics.csv
module1/results/proposed/evaluation_metrics.csv
```

Output:

```text
module1/results/comparison/baseline_vs_ocr.csv
```

Bảng chứa:

```text
Method
Ground Truth
Predicted
TP
FP
FN
Precision
Recall
F1
```

và một dòng:

```text
Difference (OCR - Baseline)
```

---

# PHẦN D — VISUALIZATION

## Step 13 — Vẽ biểu đồ so sánh

Chạy:

```powershell
python module1/proposed/13_visualize_comparison.py
```

Input:

```text
module1/results/comparison/baseline_vs_ocr.csv
```

Output:

```text
module1/results/comparison/figures/
├── precision_recall_f1.png
├── tp_fp_fn.png
└── metrics_difference.png
```

Ý nghĩa:

```text
precision_recall_f1.png
→ so trực tiếp Precision / Recall / F1

tp_fp_fn.png
→ giải thích TP / FP / FN thay đổi ra sao

metrics_difference.png
→ cho thấy OCR tăng hoặc giảm bao nhiêu so với baseline
```

---

# 4. Lệnh chạy toàn bộ pipeline từ A → Z

Nếu chạy từ đầu hoàn toàn, dùng đúng thứ tự sau.

## Baseline

```powershell
python module1/baseline/01_get_transcripts.py
python module1/baseline/02_build_documents.py
python module1/baseline/03_preprocess_documents.py
python module1/baseline/04_build_chunks.py
python module1/baseline/05_run_lda.py
python module1/baseline/06_run_lsa.py
python module1/baseline/07_run_lda_concept_matching.py
python module1/baseline/08_fuse_lda_lsa.py
python module1/baseline/09_build_video_concept_evidence.py
python module1/baseline/10_predict_evaluate_baseline.py
```

## Proposed

```powershell
python module1/proposed/02_extract_keyframes_from_youtube.py
python module1/proposed/03_ocr_keyframes.py
python module1/proposed/04_clean_ocr_text.py
python module1/proposed/05_build_transcript_ocr_chunks.py
python module1/proposed/06_run_lda_transcript_ocr.py
python module1/proposed/07_run_lsa_transcript_ocr.py
python module1/proposed/08_run_lda_concept_matching.py
python module1/proposed/09_fuse_lda_lsa.py
python module1/proposed/10_build_video_concept_evidence.py
python module1/proposed/11_predict_evaluate_proposed.py
python module1/proposed/12_compare_baseline_vs_ocr.py
python module1/proposed/13_visualize_comparison.py
```

> Không chạy:
>
> ```powershell
> python module1/proposed/01_download_videos.py
> ```

---

# 5. Pipeline tổng thể

```text
============================================================
                  MODULE 1 — BASELINE
============================================================

YouTube URLs
     ↓
Transcript
     ↓
Preprocessing
     ↓
60-second chunks
     ↓
 ┌─────────────┐
 │             │
LDA           LSA
 │             │
 └──────┬──────┘
        ↓
Fusion
0.4 LDA + 0.6 LSA
        ↓
Video-concept evidence
        ↓
Threshold = 0.40
        ↓
Prediction
        ↓
Ground Truth
        ↓
Precision / Recall / F1


============================================================
               MODULE 1 — PROPOSED
============================================================

YouTube URLs
     │
     ├──────────────→ Transcript
     │
     └→ Keyframe every 10 sec
                ↓
               OCR
                ↓
           OCR cleaning
                ↓
     merge by timestamp
                ↓
       Transcript + OCR
                ↓
          Preprocessing
                ↓
        60-second chunks
                ↓
         ┌─────────────┐
         │             │
        LDA           LSA
         │             │
         └──────┬──────┘
                ↓
              Fusion
       0.4 LDA + 0.6 LSA
                ↓
      Video-concept evidence
                ↓
         Threshold = 0.40
                ↓
            Prediction
                ↓
           Ground Truth
                ↓
      Precision / Recall / F1


============================================================
                     COMPARISON
============================================================

Transcript-only
       VS
Transcript + OCR
        ↓
baseline_vs_ocr.csv
        ↓
Visualization
```

---

# 6. Nguyên tắc để comparison công bằng

Baseline và Proposed phải giữ giống nhau ở phần downstream:

```text
chunk duration
preprocessing
LDA configuration
LSA implementation
concept catalog
subject filtering
fusion weights
prediction threshold
ground truth
evaluation metrics
```

Khác biệt chính được kiểm thử là:

```text
Baseline:
Transcript-only

Proposed:
Transcript + OCR
```

Không thay threshold riêng cho proposed chỉ để làm F1 tăng.

---

# 7. File kết quả quan trọng nhất

Sau khi chạy xong toàn bộ pipeline, các file nên kiểm tra đầu tiên là:

```text
module1/results/baseline/evaluation_metrics.csv

module1/results/proposed/evaluation_metrics.csv

module1/results/comparison/baseline_vs_ocr.csv

module1/results/comparison/figures/
├── precision_recall_f1.png
├── tp_fp_fn.png
└── metrics_difference.png
```

Đây là nhóm file dùng trực tiếp để:

```text
phân tích thực nghiệm
viết báo cáo
làm slide
giải thích với giảng viên
```

---

# 8. Nếu chỉ muốn chạy lại kết quả cuối

Nếu transcript, keyframe, OCR và intermediate files đã tồn tại, không cần chạy lại từ đầu.

Ví dụ chỉ muốn tính lại phần cuối proposed:

```powershell
python module1/proposed/09_fuse_lda_lsa.py
python module1/proposed/10_build_video_concept_evidence.py
python module1/proposed/11_predict_evaluate_proposed.py
python module1/proposed/12_compare_baseline_vs_ocr.py
python module1/proposed/13_visualize_comparison.py
```

Nếu chỉ muốn vẽ lại biểu đồ:

```powershell
python module1/proposed/13_visualize_comparison.py
```

---

# 9. Kết luận pipeline

Module 1 hiện kiểm thử câu hỏi:

```text
Việc bổ sung OCR evidence từ nội dung trực quan của video
có thay đổi khả năng phát hiện learning concepts
so với Transcript-only hay không?
```

Đánh giá bằng:

```text
Precision
Recall
F1-score
TP
FP
FN
```

Kết quả cuối được lưu ở:

```text
module1/results/comparison/baseline_vs_ocr.csv
```

và được trực quan hóa ở:

```text
module1/results/comparison/figures/
```
