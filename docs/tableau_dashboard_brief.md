# Brief: Xây dựng Dashboard Tableau — Solar PV Fault Diagnosis & Carbon Loss

> Tài liệu này cung cấp toàn bộ ngữ cảnh cần thiết để bắt đầu một cuộc hội thoại mới về việc xây dashboard Tableau. Đọc hết phần 1-3 trước khi tư vấn.

---

## 1. Bối cảnh dự án

### 1.1. Dự án là gì
Một nghiên cứu (đã viết thành bài báo, chuẩn bị nộp hội nghị EAI FISAT 2026) về **chẩn đoán lỗi inverter điện mặt trời và quy đổi tổn thất carbon**. Pipeline Python đã hoàn tất, đã audit kỹ, tái lập được 100%.

### 1.2. Bài toán gốc
Nhà máy điện mặt trời có nhiều inverter. Khi một inverter hỏng/suy giảm, sản lượng tụt — nhưng khó phát hiện vì sản lượng vốn dao động theo thời tiết. Hệ thống này:
1. Dùng model XGBoost (global, một model cho cả nhà máy) dự đoán **"công suất AC đáng lẽ phải có"** dựa trên điều kiện thời tiết.
2. So sánh thực tế với kỳ vọng → phân loại lỗi theo 3 lớp.
3. Quy đổi tổn thất thành kWh và tấn CO₂.

### 1.3. Logic phân loại lỗi (taxonomy) — quan trọng để hiểu dashboard
| Lớp | Điều kiện | Hành động O&M |
|---|---|---|
| `TOTAL_LOSS` | AC power = 0 **và** irradiation > 0.2 (có nắng mà không ra điện) | Điều đội sửa ngay |
| `PARTIAL_LOSS` | 0 < AC power < 50% trung bình các inverter cùng nhà máy, và irradiation > 0.2 | Kiểm tra trong 24h |
| `NORMAL` | Còn lại (bao gồm toàn bộ ban đêm) | Không cần hành động |

### 1.4. Khái niệm "Masked Loss" — cần hiểu để không vẽ sai
- **Raw loss** (tổn thất thô) = cộng dồn `max(expected − actual, 0)` trên mọi thời điểm → ra **847,213 kWh**. Con số này **SAI** (bị phồng do nhiễu model tích lũy một chiều).
- **Masked loss** (tổn thất đã lọc) = chỉ tính tổn thất tại các thời điểm taxonomy xác nhận có lỗi → ra **639,306 kWh**. Đây là con số **ĐÚNG** dùng trong báo cáo.
- Dashboard phải dùng **masked loss**, không dùng raw loss.

---

## 2. Dữ liệu có sẵn

### 2.1. File dữ liệu chính
Pipeline sinh ra file: `data/processed/paper/paper_dataset_with_masked_loss.csv`

Đây là file **master** chứa mọi thứ cần cho dashboard (136,476 dòng, đo mỗi 15 phút).

### 2.2. Các cột chính (cần verify lại tên chính xác khi mở file)
| Cột | Ý nghĩa |
|---|---|
| `DATE_TIME` | Thời điểm đo (15 phút/lần) |
| `PLANT_NAME` | Plant 1 / Plant 2 |
| `SOURCE_KEY` | Mã inverter (ví dụ `bvBOhCH3iADSZry`) |
| `AC_POWER` | Công suất AC thực tế (kW) |
| `DC_POWER_CORRECTED` | Công suất DC (đã sửa lỗi thang đo) |
| `EXPECTED_AC_POWER` | Công suất AC **kỳ vọng** (output của model XGBoost) |
| `ANOMALY_CLASS` | NORMAL / TOTAL_LOSS / PARTIAL_LOSS |
| `TRUE_ENERGY_LOSS_KWH` | Tổn thất năng lượng đã mask (kWh) — **dùng cột này** |
| `POWER_LOSS_KW` | Chênh lệch công suất tức thời |
| `ENERGY_KWH_INTERVAL` | Năng lượng thực tế sinh ra trong khoảng 15 phút |
| `IRRADIATION` | Bức xạ mặt trời |
| `MODULE_TEMPERATURE` | Nhiệt độ tấm pin |
| `AMBIENT_TEMPERATURE` | Nhiệt độ môi trường |
| `IS_DAY` | True/False — ban ngày hay ban đêm |
| `POWER_GAP_PERCENT` | % chênh lệch so với peer |
| `EFFICIENCY_CORRECTED` | Hiệu suất DC→AC |

**Lưu ý:** khi bắt đầu làm, nên mở file kiểm tra tên cột thực tế trước — danh sách trên lấy từ ngữ cảnh dự án, có thể có sai lệch nhỏ.

### 2.3. Phạm vi dữ liệu
- **Thời gian**: 15/05/2020 – 17/06/2020 (34 ngày)
- **2 nhà máy**: Plant 1 (vận hành tốt), Plant 2 (có nhiều inverter suy giảm)
- **Tần suất**: 15 phút/lần
- Dữ liệu gốc: Kaggle "Solar Power Generation Data" (Ani Kannal), 2 nhà máy ở Ấn Độ

---

## 3. Các con số chuẩn (dashboard phải khớp)

Đây là kết quả đã được audit kỹ — mọi con số trên dashboard phải trùng khớp:

### Tổn thất năng lượng & carbon
| Chỉ số | Plant 1 | Plant 2 | Tổng |
|---|---|---|---|
| Masked Energy Loss (kWh) | 16,452.13 | 622,853.97 | **639,306.10** |
| CO₂ Loss (kg) | 11,565.85 | 437,866.34 | **449,432.19** |
| Total Actual Energy (kWh) | 5,291,787.50 | 4,083,502.38 | — |
| Loss Rate (%) | 0.31% | 15.25% | — |

- Hệ số phát thải: **0.703 kgCO₂/kWh** (CEA India, FY2020-21)
- Bootstrap 95% CI cho tổng tổn thất: 525k – 748k kWh

### Phân bố taxonomy
| Lớp | Toàn bộ dataset | Chỉ ban ngày |
|---|---|---|
| NORMAL | 132,282 (96.93%) | 72,904 |
| TOTAL_LOSS | 3,821 (2.80%) | 3,821 |
| PARTIAL_LOSS | 373 (0.27%) | 373 |

**Lưu ý quan trọng**: NORMAL khác nhau giữa 2 cột vì ban đêm mặc định là NORMAL. Dashboard nên **lọc ban ngày** (`IS_DAY = True`) cho hầu hết phân tích, trừ khi cố ý muốn xem toàn bộ.

### Phân bố anomaly theo Plant
- Plant 2 chiếm **>98%** tổng số lỗi (TOTAL_LOSS: 3,758 / PARTIAL_LOSS: 356)
- Plant 1 rất ít lỗi (TOTAL_LOSS: 63 / PARTIAL_LOSS: 17)

### Tổn thất theo Plant × Class
| Plant | Class | Energy Loss (kWh) |
|---|---|---|
| Plant 1 | PARTIAL_LOSS | 2,585.45 |
| Plant 1 | TOTAL_LOSS | 13,866.68 |
| Plant 2 | PARTIAL_LOSS | 41,294.28 |
| Plant 2 | TOTAL_LOSS | 581,559.68 |

### Model performance (tham khảo — KHÔNG cần lên Tableau, thuộc phần Streamlit sau này)
| Plant | Split | R² | MAE (kW) | RMSE (kW) | MAPE (%) |
|---|---|---|---|---|---|
| Plant 1 | Train | 0.9775 | 29.10 | 58.00 | 6.03 |
| Plant 1 | Test | 0.9619 | 31.08 | 68.59 | 9.24 |
| Plant 2 | Train | 0.5403 | 156.72 | 279.36 | 22.18 |
| Plant 2 | Test | 0.5386 | 101.45 | 199.23 | 33.38 |

**Giải thích R² Plant 2 thấp**: đây KHÔNG phải model tệ. Global model cố tình không fit khít từng inverter, để inverter suy giảm lộ ra. R² thấp = đang phơi bày degradation thật ở Plant 2.

---

## 4. Mục tiêu dashboard

### 4.1. Quyết định đã chốt
- **Công cụ**: Tableau
- **Góc nhìn**: mô phỏng **người vận hành thật (O&M operator)** — không phải dashboard học thuật. Trọng tâm là *hành động*: inverter nào cần sửa, ưu tiên cái nào, mất bao nhiêu.
- **Quy mô**: **nhiều tab** (multi-tab), có drill-down từ tổng quan xuống chi tiết.

### 4.2. Phạm vi TÁCH RIÊNG (làm sau, không thuộc Tableau)
Phần trình bày **phương pháp luận và model** sẽ được làm bằng **Streamlit**, tích hợp API AI để trả lời câu hỏi về phương pháp/model một cách tương tác.
→ Nghĩa là: **Tableau KHÔNG cần** hiển thị R², MAE, SHAP, so sánh baseline UML, hay giải thích global-vs-local. Những thứ đó thuộc app Streamlit sau này.
→ Tableau chỉ tập trung vào **vận hành**: lỗi ở đâu, nghiêm trọng thế nào, mất bao nhiêu, làm gì tiếp.

### 4.3. Đề xuất cấu trúc tab (khởi điểm để thảo luận)

**Tab 1 — Operations Overview (màn hình chính hàng ngày)**
- KPI cards: tổng tổn thất kWh, tổng CO₂, loss rate %, số inverter đang có lỗi
- Cảnh báo đang mở: bao nhiêu TOTAL_LOSS (dispatch ngay), bao nhiêu PARTIAL_LOSS (kiểm tra 24h)
- So sánh nhanh Plant 1 vs Plant 2
- Timeline tổng: lỗi phân bố theo ngày

**Tab 2 — Alert Queue / Danh sách công việc**
- Bảng inverter cần xử lý, sắp xếp theo mức ưu tiên (tổn thất kWh giảm dần)
- Mỗi dòng: mã inverter, plant, loại lỗi, thời điểm, tổn thất kWh, CO₂, hành động khuyến nghị
- Filter theo plant / loại lỗi / khoảng thời gian
- Đây là tab "để làm việc" — người vận hành mở ra biết đi sửa cái nào trước

**Tab 3 — Inverter Detail (drill-down)**
- Chọn 1 inverter → xem timeline Actual vs Expected AC Power
- Đánh dấu các điểm TOTAL_LOSS (đỏ) / PARTIAL_LOSS (cam)
- Biểu đồ tổn thất công suất theo thời gian
- Thống kê: tổng tổn thất của inverter này, số sự kiện, thời gian downtime
- (Case study tham khảo: inverter `bvBOhCH3iADSZry` ở Plant 1 có 2 sự kiện TOTAL_LOSS rõ rệt ~08/06 và ~14/06/2020)

**Tab 4 — Loss & Carbon Impact**
- Phân rã tổn thất: theo plant, theo loại lỗi, theo thời gian
- Quy đổi CO₂ (hệ số 0.703)
- Xếp hạng inverter theo tổn thất (top N tệ nhất)
- Ghi chú rõ đây là **ước lượng cận trên** (xem Mục 5.4)

**Tab 5 (tuỳ chọn) — Plant Comparison**
- So sánh sâu Plant 1 vs Plant 2: tỷ lệ lỗi, phân bố inverter, loss rate
- Cho thấy Plant 2 chiếm >98% tổng lỗi

### 4.4. Bối cảnh kỹ thuật
- Người dùng đã từng có file Power BI (`power_bi/dashboard.pbix`) nhưng giờ chuyển sang **Tableau**
- Chưa rõ trình độ Tableau của người dùng — nên hỏi trước
- Chưa rõ Tableau Desktop hay Tableau Public (ảnh hưởng tới việc publish/chia sẻ)

## 5. Những điều CẦN LƯU Ý khi tư vấn

### 5.1. Đừng vẽ tổn thất sai cách
- Dùng `TRUE_ENERGY_LOSS_KWH` (đã mask), **không** tự tính `expected − actual` rồi cộng dồn (sẽ ra raw loss 847k, sai).
- Nếu cần tính CO₂: nhân masked loss với 0.703.

### 5.2. Cẩn thận với ban đêm
Ban đêm AC = 0 là **bình thường**, không phải lỗi. Taxonomy đã xử lý (yêu cầu `irradiation > 0.2`), nhưng nếu dashboard tự tính toán gì đó thì phải lọc `IS_DAY = True`, nếu không sẽ ra kết quả vô nghĩa.

### 5.3. Loss Rate tính theo Actual, không phải Expected
Công thức đúng: `Loss Rate = Masked Loss / Total Actual Energy`. Đây là điều đã được xác minh kỹ trong quá trình audit (từng nhầm dùng Expected Energy → ra số sai).

### 5.4. Đừng over-claim trên dashboard
Con số 639 MWh là **ước lượng cận trên** (upper-bound), không phải tổn thất đã được xác thực. Bài báo đã thừa nhận điều này. Nếu dashboard hiển thị nó như "số tiền chắc chắn thu hồi được" thì sai lệch. Nên có chú thích.

### 5.5. Inverter đáng chú ý
`bvBOhCH3iADSZry` (Plant 1) là inverter được dùng làm case study trong bài — có 2 sự kiện TOTAL_LOSS rõ rệt (khoảng 08/06 và 14/06/2020).

---

## 6. Việc cần làm trong cuộc hội thoại mới

1. **Hỏi rõ yêu cầu**: người dùng muốn dashboard phục vụ ai (kỹ thuật viên? quản lý? hội đồng chấm bài?), bao nhiêu trang, mức độ phức tạp.
2. **Thiết kế cấu trúc dashboard**: chia trang/tab, xác định biểu đồ nào ở đâu.
3. **Hướng dẫn chuẩn bị dữ liệu**: file CSV cần xử lý gì trước khi đưa vào Tableau (aggregate sẵn? giữ nguyên 136k dòng?).
4. **Hướng dẫn từng bước dựng trong Tableau**: calculated fields, filters, parameters, actions (drill-down), dashboard layout.
5. **Kiểm tra tính đúng đắn**: đối chiếu số trên dashboard với bảng ở Mục 3.

---

## 7. Tình trạng dự án hiện tại (để tham khảo)

- Pipeline Python: hoàn tất, modular (`src/`), chạy bằng `python reproduce_all.py`
- Code công khai tại GitHub: `github.com/anle26/Solar_Uni` (đã dọn sạch, không chứa data)
- Bài báo: đã sửa theo phản hồi reviewer, chuẩn bị nộp EAI FISAT 2026 (Scopus Q4, tháng 11/2026)
- Dashboard Tableau (góc nhìn O&M, multi-tab): **chưa bắt đầu** — đây là việc cần làm ngay
- App Streamlit + AI API giải thích phương pháp/model: **kế hoạch tương lai**, chưa bắt đầu, tách riêng khỏi Tableau
