# Brief: Xây dựng Streamlit App — Solar PV Fault Diagnosis & Carbon Loss

> Tài liệu cung cấp toàn bộ ngữ cảnh để bắt đầu một cuộc hội thoại mới về việc xây app Streamlit. Đọc hết Mục 1–4 trước khi tư vấn kỹ thuật.

---

## 0. TÓM TẮT YÊU CẦU

Xây **một app Streamlit duy nhất, nhiều trang (multipage)**, gồm 2 phần:

| Phần | Mục đích | Người xem |
|---|---|---|
| **A. Operations Dashboard** | Mô phỏng góc nhìn người vận hành nhà máy (O&M): lỗi ở đâu, sửa cái nào trước, mất bao nhiêu kWh/CO₂ | Kỹ thuật viên vận hành |
| **B. Methodology Explainer** | Giải thích phương pháp & model, tích hợp **API AI** để trả lời câu hỏi tương tác | Giảng viên / hội đồng / người đọc bài báo |

**Không dùng Tableau** (đã cân nhắc và loại bỏ — Streamlit làm được cả 2 phần, lại truy cập trực tiếp được model Python).

---

## 1. Bối cảnh dự án

### 1.1. Dự án là gì
Nghiên cứu về **chẩn đoán lỗi inverter điện mặt trời và quy đổi tổn thất carbon**, đã viết thành bài báo chuẩn bị nộp hội nghị EAI FISAT 2026 (Scopus Q4). Pipeline Python đã hoàn tất, đã audit kỹ, tái lập được 100%.

### 1.2. Bài toán gốc
Nhà máy điện mặt trời có nhiều inverter. Khi một inverter hỏng/suy giảm, sản lượng tụt — nhưng khó phát hiện vì sản lượng vốn dao động theo thời tiết. Hệ thống này:
1. Dùng **XGBoost** dự đoán *"công suất AC đáng lẽ phải có"* dựa trên điều kiện thời tiết.
2. So sánh thực tế với kỳ vọng → phân loại lỗi 3 lớp.
3. Quy đổi tổn thất thành kWh và tấn CO₂.

### 1.3. Ba quyết định thiết kế cốt lõi (cần hiểu để giải thích ở phần B)

**(a) Loại DC power khỏi feature set — tránh target leakage.**
Nếu cho model biết DC power, nó chỉ học `AC ≈ 0.98 × DC` → khi cụm pin bị che bóng, DC tụt thì model dự đoán AC cũng tụt → sai số = 0 → không phát hiện được lỗi. Nên model chỉ dùng 5 biến thời tiết/thời gian: `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `HOUR_SIN`, `HOUR_COS`.

**(b) Global model thay vì Local model — chống anomaly masking.**
- *Local* (mỗi inverter một model): fit khít từng inverter → học luôn trạng thái suy giảm thành "bình thường" → không phát hiện được lỗi. Nhóm Nahar et al. dùng cách này, R² = 0.98 / 0.91.
- *Global* (một model cho cả nhà máy): inverter suy giảm bị đo so với chuẩn chung → lộ ra. Bài này dùng cách này, R² = 0.9619 / 0.5386.
- **R² thấp ở Plant 2 KHÔNG phải model tệ** — đó là model đang phơi bày degradation thật.

**(c) Masked Loss Aggregation — chống Positive Drift Bias.**
Cộng dồn `max(expected − actual, 0)` trên mọi thời điểm → nhiễu ngẫu nhiên bị tích lũy một chiều → ra **847,213 kWh** (SAI, phồng). Chỉ tính tổn thất khi taxonomy xác nhận có lỗi → ra **639,306 kWh** (ĐÚNG).

### 1.4. Logic phân loại lỗi (taxonomy)
| Lớp | Điều kiện | Hành động O&M |
|---|---|---|
| `TOTAL_LOSS` | AC power = 0 **và** irradiation > 0.2 | Điều đội sửa ngay |
| `PARTIAL_LOSS` | 0 < AC power < 50% trung bình inverter cùng nhà máy, và irradiation > 0.2 | Kiểm tra trong 24h |
| `NORMAL` | Còn lại (gồm toàn bộ ban đêm) | Không cần hành động |

---

## 2. Tài nguyên có sẵn (Streamlit truy cập trực tiếp được)

### 2.1. Dữ liệu
File master: `data/processed/paper/paper_dataset_with_masked_loss.csv` (136,476 dòng, 15 phút/lần)

| Cột | Ý nghĩa |
|---|---|
| `DATE_TIME` | Thời điểm đo |
| `PLANT_NAME` | Plant 1 / Plant 2 |
| `SOURCE_KEY` | Mã inverter (vd `bvBOhCH3iADSZry`) |
| `AC_POWER` | Công suất AC thực tế (kW) |
| `DC_POWER_CORRECTED` | Công suất DC (đã sửa thang đo) |
| `EXPECTED_AC_POWER` | Công suất AC kỳ vọng (output model) |
| `ANOMALY_CLASS` | NORMAL / TOTAL_LOSS / PARTIAL_LOSS |
| `TRUE_ENERGY_LOSS_KWH` | Tổn thất đã mask — **dùng cột này** |
| `POWER_LOSS_KW` | Chênh lệch công suất tức thời |
| `ENERGY_KWH_INTERVAL` | Năng lượng thực tế trong 15 phút |
| `IRRADIATION`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE` | Biến thời tiết |
| `IS_DAY` | True/False |
| `POWER_GAP_PERCENT` | % chênh lệch so với peer |
| `EFFICIENCY_CORRECTED` | Hiệu suất DC→AC |

*Lưu ý: nên mở file kiểm tra tên cột thực tế trước khi code.*

### 2.2. Model đã train (lợi thế lớn của Streamlit so với Tableau)
- `models/expected_power_model_4135001.json` (Plant 1)
- `models/expected_power_model_4136001.json` (Plant 2)

Load bằng `xgboost.XGBRegressor().load_model(...)` → **chạy dự đoán trực tiếp trong app**, làm được:
- What-if simulation: người dùng chỉnh irradiation/nhiệt độ → xem expected power thay đổi
- SHAP interactive: giải thích từng dự đoán cụ thể
- So sánh dự đoán vs thực tế theo yêu cầu người dùng

### 2.3. Code pipeline (import lại được)
```
src/preprocessing.py          # xử lý dữ liệu thô, sửa thang đo DC
src/expected_power_model.py   # train XGBoost
src/taxonomy.py               # phân loại lỗi (apply_taxonomy)
src/masked_loss.py            # tính masked loss
src/carbon_quantification.py  # quy đổi CO2
src/baselines.py              # so sánh IF/OCSVM/LOF
src/fault_injection.py        # tạo lỗi giả để đánh giá
figures/generate_all_figures.py
reproduce_all.py              # entrypoint chạy toàn bộ
config.yaml                   # hyperparameters, emission factor
```

### 2.4. Kết quả đánh giá (dùng cho phần B)
| Method | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|
| Rule-Based Taxonomy | 1.00 ± 0.00 | 0.58 ± 0.06 | 0.74 ± 0.05 | 0.74 ± 0.05 |
| Isolation Forest | 0.32 ± 0.04 | 0.29 ± 0.02 | 0.30 ± 0.02 | 0.30 ± 0.02 |
| One-Class SVM | 0.17 ± 0.04 | 0.52 ± 0.11 | 0.25 ± 0.03 | 0.24 ± 0.03 |
| LOF | 0.15 ± 0.06 | 0.39 ± 0.10 | 0.21 ± 0.05 | 0.21 ± 0.05 |

Model performance:
| Plant | Split | R² | MAE (kW) | RMSE (kW) | MAPE (%) |
|---|---|---|---|---|---|
| Plant 1 | Train | 0.9775 | 29.10 | 58.00 | 6.03 |
| Plant 1 | Test | 0.9619 | 31.08 | 68.59 | 9.24 |
| Plant 2 | Train | 0.5403 | 156.72 | 279.36 | 22.18 |
| Plant 2 | Test | 0.5386 | 101.45 | 199.23 | 33.38 |

SHAP feature ranking: `IRRADIATION > MODULE_TEMPERATURE > HOUR_COS > AMBIENT_TEMPERATURE > HOUR_SIN`

---

## 3. Các con số chuẩn (app phải khớp)

### Tổn thất năng lượng & carbon
| Chỉ số | Plant 1 | Plant 2 | Tổng |
|---|---|---|---|
| Masked Energy Loss (kWh) | 16,452.13 | 622,853.97 | **639,306.10** |
| CO₂ Loss (kg) | 11,565.85 | 437,866.34 | **449,432.19** |
| Total Actual Energy (kWh) | 5,291,787.50 | 4,083,502.38 | — |
| Loss Rate (%) | 0.31% | 15.25% | — |

- Hệ số phát thải: **0.703 kgCO₂/kWh** (CEA India, FY2020-21)
- Bootstrap 95% CI tổng tổn thất: 525k – 748k kWh
- Raw loss (trước mask): 847,213.65 kWh — **KHÔNG dùng con số này để báo cáo**

### Phân bố taxonomy
| Lớp | Toàn bộ | Chỉ ban ngày |
|---|---|---|
| NORMAL | 132,282 (96.93%) | 72,904 |
| TOTAL_LOSS | 3,821 (2.80%) | 3,821 |
| PARTIAL_LOSS | 373 (0.27%) | 373 |

### Phân bố theo Plant
- Plant 2 chiếm **>98%** tổng lỗi: TOTAL_LOSS 3,758 / PARTIAL_LOSS 356
- Plant 1: TOTAL_LOSS 63 / PARTIAL_LOSS 17

### Tổn thất theo Plant × Class
| Plant | Class | Energy Loss (kWh) |
|---|---|---|
| Plant 1 | PARTIAL_LOSS | 2,585.45 |
| Plant 1 | TOTAL_LOSS | 13,866.68 |
| Plant 2 | PARTIAL_LOSS | 41,294.28 |
| Plant 2 | TOTAL_LOSS | 581,559.68 |

### Phạm vi dữ liệu
- 15/05/2020 – 17/06/2020 (34 ngày), 2 nhà máy ở Ấn Độ
- Nguồn: Kaggle "Solar Power Generation Data" (Ani Kannal)
- Train/test split: **theo ngày** (27 ngày đầu train, 7 ngày cuối test, mốc 11/06/2020)
- Mỗi nhà máy có model riêng; "global" nghĩa là *một model cho mọi inverter trong cùng nhà máy*
- Inverter `bvBOhCH3iADSZry` bị **cố ý loại khỏi tập train** (để baseline không học trạng thái hỏng của nó)

---

## 4. Cấu trúc app đề xuất

### PHẦN A — Operations Dashboard (ưu tiên làm trước)

**Trang 1: Operations Overview**
- KPI cards: tổng tổn thất kWh, tổng CO₂ (tấn), loss rate %, số inverter đang có lỗi
- Cảnh báo: bao nhiêu TOTAL_LOSS (dispatch ngay) / PARTIAL_LOSS (kiểm tra 24h)
- So sánh nhanh Plant 1 vs Plant 2
- Timeline tổng: phân bố lỗi theo ngày

**Trang 2: Alert Queue (tab quan trọng nhất)**
- Bảng inverter cần xử lý, sắp xếp theo ưu tiên (tổn thất kWh giảm dần)
- Cột: mã inverter, plant, loại lỗi, thời điểm, tổn thất kWh, CO₂, hành động khuyến nghị
- Filter: plant / loại lỗi / khoảng thời gian
- **Vấn đề thiết kế cần bàn**: dữ liệu là 34 ngày lịch sử, không real-time. Hai cách xử lý:
  - (i) Dùng slider chọn "ngày hiện tại" → mô phỏng như đang vận hành ngày đó (thật hơn)
  - (ii) Gộp toàn bộ 34 ngày thành danh sách sự kiện đã xảy ra (đơn giản hơn)

**Trang 3: Inverter Detail (drill-down)**
- Chọn 1 inverter → timeline Actual vs Expected AC Power
- Đánh dấu TOTAL_LOSS (đỏ) / PARTIAL_LOSS (cam)
- Biểu đồ tổn thất công suất theo thời gian
- Thống kê: tổng tổn thất, số sự kiện, thời gian downtime
- Case study tham khảo: `bvBOhCH3iADSZry` (Plant 1) có 2 sự kiện TOTAL_LOSS rõ rệt ~08/06 và ~14/06/2020

**Trang 4: Loss & Carbon Impact**
- Phân rã tổn thất: theo plant, theo loại lỗi, theo thời gian
- Xếp hạng inverter theo tổn thất (top N tệ nhất)
- Quy đổi CO₂
- **Ghi chú bắt buộc**: đây là *ước lượng cận trên* (xem Mục 5.4)

### PHẦN B — Methodology Explainer (làm sau)

**Trang 5: How It Works**
- Giải thích trực quan pipeline 7 bước
- Minh họa target leakage: tại sao loại DC power (có thể demo bằng cách so sánh model có/không có DC)
- Minh họa anomaly masking: global vs local

**Trang 6: Model & Performance**
- Bảng R²/MAE/RMSE/MAPE
- Scatter Actual vs Predicted
- SHAP feature importance (chạy live từ model đã load)
- **What-if simulator**: slider chỉnh irradiation/nhiệt độ → xem expected power thay đổi
- Giải thích rõ vì sao R² Plant 2 = 0.54 lại là điều tốt

**Trang 7: Evaluation & Baselines**
- Bảng so sánh Rule-Based vs IF/OCSVM/LOF
- PR curves
- Giải thích synthetic fault injection
- **Trung thực nêu limitation**: circularity (luật TOTAL_LOSS trùng cơ chế inject fault), chưa có fault thật được validate

**Trang 8: AI Assistant (tích hợp API)**
- Chatbot trả lời câu hỏi về phương pháp, model, kết quả
- Cần: system prompt chứa ngữ cảnh dự án (có thể dùng chính file brief này làm nền)
- Cân nhắc: RAG trên nội dung bài báo (`docs/paper/main.tex`), hoặc nhồi context tĩnh
- Lưu ý bảo mật: **không hardcode API key** trong code — dùng `st.secrets` hoặc biến môi trường

---

## 5. Cạm bẫy cần tránh (quan trọng)

### 5.1. Đừng tính tổn thất sai cách
Dùng `TRUE_ENERGY_LOSS_KWH` (đã mask). **Không** tự tính `expected − actual` rồi cộng dồn → sẽ ra 847k (raw loss, sai lệch 33%).

### 5.2. Cẩn thận ban đêm
Ban đêm AC = 0 là bình thường. Nếu app tự tính toán gì đó, phải lọc `IS_DAY = True`, nếu không kết quả vô nghĩa.

### 5.3. Loss Rate tính theo Actual, không phải Expected
Công thức đúng: `Loss Rate = Masked Loss / Total Actual Energy` (dùng `ENERGY_KWH_INTERVAL`). Đây là lỗi đã từng xảy ra trong quá trình audit — dùng nhầm Expected Energy cho ra số sai.

### 5.4. Đừng over-claim
639 MWh là **ước lượng cận trên (upper-bound)**, không phải tổn thất đã xác thực. Global model không phân biệt được "inverter hỏng" với "inverter khác biệt hợp lệ" (vị trí, công suất, đấu nối khác nhau). Chưa có nhật ký O&M nào đối chứng. App nên có chú thích rõ.

### 5.5. Train/test có bản chất khác nhau
`EXPECTED_AC_POWER` ở phần train là in-sample (model đã thấy), phần test là out-of-sample. Cả 2 đều dùng để tính tổn thất — chấp nhận được cho mục đích vận hành, nhưng nên biết để trả lời nếu bị hỏi. Có thể thêm filter xem riêng giai đoạn test.

### 5.6. Hiệu năng Streamlit với 136k dòng
- Dùng `@st.cache_data` cho việc load CSV, `@st.cache_resource` cho load model
- Cân nhắc pre-aggregate sẵn các bảng tổng hợp (theo ngày, theo inverter) thay vì group by mỗi lần render
- Nếu chậm: cân nhắc chuyển CSV sang Parquet

---

## 6. Việc cần làm trong cuộc hội thoại mới

1. **Hỏi rõ**: trình độ Python/Streamlit của người dùng, deploy ở đâu (local / Streamlit Cloud / server), có ràng buộc thời gian không.
2. **Chốt cách xử lý "thời gian"** cho Alert Queue (mô phỏng real-time hay tổng hợp lịch sử) — quyết định này ảnh hưởng toàn bộ Phần A.
3. **Thiết kế cấu trúc thư mục** app (multipage Streamlit: `pages/` folder).
4. **Xây Phần A trước, từng trang một**, kiểm chứng số liệu với Mục 3 sau mỗi trang.
5. **Phần B làm sau**, đặc biệt trang AI Assistant làm cuối cùng.
6. **Không commit API key** — hướng dẫn dùng `.streamlit/secrets.toml` và thêm vào `.gitignore`.

---

## 7. Tình trạng dự án hiện tại

- Pipeline Python: hoàn tất, modular (`src/`), chạy bằng `python reproduce_all.py`, đã verify deterministic 3 lần
- Code công khai: `github.com/anle26/Solar_Uni` (đã dọn sạch, **không chứa data** — data tải từ Kaggle theo hướng dẫn trong README)
- Bài báo: đã sửa theo phản hồi reviewer, chuẩn bị nộp EAI FISAT 2026
- Đã có file Power BI cũ (`power_bi/dashboard.pbix`) — không còn dùng
- **Streamlit app: chưa bắt đầu — đây là việc mới**
