# Tableau Build Guide — Solar PV Fault Diagnosis & Carbon Loss Dashboard

> Kế thừa từ [tableau_dashboard_brief.md](tableau_dashboard_brief.md). File này là hướng dẫn triển khai cụ thể trong Tableau Public, dùng calculated field syntax thật.

## 0. Xác minh dữ liệu (đã làm)

Đã mở `data/processed/paper/paper_dataset_with_masked_loss.csv` (136,476 dòng) và đối chiếu với bảng số ở Mục 3 của brief:

| Chỉ số | Brief | File thực tế | Khớp? |
|---|---|---|---|
| Masked Loss Plant 1 / Plant 2 / Tổng | 16,452.13 / 622,853.97 / 639,306.10 | 16,452.13 / 622,853.97 / 639,306.10 | ✅ |
| CO₂ Loss (dùng cột `TRUE_CO2_LOSS_KG` có sẵn) | 11,565.85 / 437,866.34 / 449,432.19 | 11,565.85 / 437,866.34 / 449,432.19 | ✅ |
| Loss Rate % (Plant 1 / Plant 2) | 0.31% / 15.25% | 0.3109% / 15.2529% | ✅ |
| Taxonomy count (toàn bộ / chỉ ngày) | khớp bảng Mục 3 | khớp | ✅ |
| Plant × Class breakdown | khớp bảng Mục 3 | khớp | ✅ |
| Case study inverter `bvBOhCH3iADSZry` | Plant 1, 2 sự kiện TOTAL_LOSS ~08/06 và ~14/06 | Plant 1, sự kiện đầu bắt đầu 07/06 12:15 (kéo dài qua 08/06) | ✅ (lệch nhẹ do biên sự kiện, không phải sai số) |
| Total Actual Energy Plant 1 | 5,291,787.50 | 5,292,514.42 | ⚠️ lệch 727 kWh (~0.014%) |

**Lưu ý về dòng ⚠️**: chênh lệch cực nhỏ, không ảnh hưởng Loss Rate % (vẫn khớp làm tròn 0.31%). Có thể do bài báo làm tròn/lọc khác biệt nhỏ ở bước tổng hợp cuối. Không chặn việc build dashboard — nhưng nếu bạn muốn dashboard khớp 100% với số trong bài báo đã nộp, nên dùng SUM(`ENERGY_KWH_INTERVAL`) trực tiếp từ Tableau (con số 5,292,514.42) làm chuẩn, vì đây là tính trực tiếp từ file gốc, thay vì gõ cứng số 5,291,787.50 từ bài báo.

**Column mapping thực tế** (khác brief một chút — dùng tên này khi kéo thả):
- Không có sẵn cột tên chung "carbon 0.703" — **đã có sẵn** `TRUE_CO2_LOSS_KG` (= `TRUE_ENERGY_LOSS_KWH` × 0.703 tính sẵn), dùng trực tiếp, không cần tự nhân.
- `PLANT_ID` cũng tồn tại song song `PLANT_NAME` — dùng `PLANT_NAME` cho hiển thị.
- `DATE_TIME` là string dạng `YYYY-MM-DD HH:MM:SS` — Tableau sẽ tự nhận là Date & Time khi import, nhưng **kiểm tra lại** sau khi kéo vào (đổi Data Type nếu bị nhận nhầm thành String).

---

## 1. Kết nối dữ liệu trong Tableau Public

1. Tableau Public → **Connect to Data** → Text File → chọn `paper_dataset_with_masked_loss.csv`.
2. Kéo bảng vào canvas, Tableau Public bắt buộc **Extract** (không có Live cho file text) — để nguyên mặc định.
3. Trong Data Source tab, kiểm tra:
   - `DATE_TIME` → Date & Time
   - `IS_DAY` → Boolean
   - `ANOMALY_CLASS`, `PLANT_NAME`, `SOURCE_KEY` → String/Dimension
   - Các cột `*_KWH`, `*_KG`, `*_KW`, `*_POWER` → Number (Decimal), Measure
4. Không cần tách file hay pre-aggregate trước — 136k dòng nhẹ, Tableau xử lý được trực tiếp bằng calculated field + LOD expression. Giữ 1 nguồn dữ liệu duy nhất = tránh lệch số giữa các tab.

---

## 2. Calculated Fields cần tạo (tạo hết trước khi build sheet)

### 2.1. Nhóm cơ bản

```
// Recommended Action — dùng cho Tab 2, Tab 3
Recommended Action
CASE [ANOMALY_CLASS]
WHEN "TOTAL_LOSS" THEN "Điều đội sửa ngay"
WHEN "PARTIAL_LOSS" THEN "Kiểm tra trong 24h"
ELSE "Không cần hành động"
END
```

```
// Severity Order — dùng để sort/tô màu nhất quán (đỏ > cam > xám)
Severity Order
CASE [ANOMALY_CLASS]
WHEN "TOTAL_LOSS" THEN 1
WHEN "PARTIAL_LOSS" THEN 2
ELSE 3
END
```

```
// Is Fault — đếm số bản ghi có lỗi (không phải số inverter)
Is Fault
IF [ANOMALY_CLASS] <> "NORMAL" THEN 1 ELSE 0 END
```

```
// Loss Rate % — LUÔN tính theo Actual, KHÔNG theo Expected (mục 5.3 của brief)
Loss Rate %
SUM([TRUE_ENERGY_LOSS_KWH]) / SUM([ENERGY_KWH_INTERVAL])
```
→ Format số: Percentage, 2 chữ số thập phân.

### 2.2. Nhóm cấp inverter (FIXED LOD) — dùng cho Tab 2 Alert Queue

```
// Worst severity mà inverter này từng có, trong toàn bộ khoảng thời gian đang filter
Inverter Worst Severity
{FIXED [SOURCE_KEY] : MIN([Severity Order])}
```

```
// Recommended Action ở cấp inverter (dùng Inverter Worst Severity thay vì Severity Order dòng lẻ)
Inverter Recommended Action
CASE [Inverter Worst Severity]
WHEN 1 THEN "Điều đội sửa ngay"
WHEN 2 THEN "Kiểm tra trong 24h"
ELSE "Không cần hành động"
END
```

```
// Tổng tổn thất kWh của từng inverter — dùng để sort Alert Queue giảm dần
Inverter Total Loss (kWh)
{FIXED [SOURCE_KEY] : SUM([TRUE_ENERGY_LOSS_KWH])}
```

```
// Số sự kiện lỗi (đếm dòng 15 phút có lỗi) của inverter — không phải số "sự cố" thực,
// chỉ là số điểm đo bị gắn nhãn lỗi. Ghi rõ nhãn "Fault Records" trên UI, không gọi là "events"
// để tránh hiểu nhầm là số lần hỏng.
Inverter Fault Record Count
{FIXED [SOURCE_KEY] : SUM([Is Fault])}
```

### 2.3. Ngày/đêm

Dùng trực tiếp field `IS_DAY` (Boolean có sẵn) làm Filter — không cần calculated field riêng. Áp dụng `IS_DAY = True` cho **mọi sheet hiển thị phân bố/đếm theo ANOMALY_CLASS** (Tab 1 breakdown, Tab 5). Không cần áp cho các KPI tổng tổn thất kWh/CO₂ vì lỗi chỉ xảy ra ban ngày theo định nghĩa taxonomy (irradiation > 0.2), nên KPI không đổi dù có lọc IS_DAY hay không — nhưng lọc vào vẫn an toàn và nhất quán, khuyến nghị bật mặc định trên toàn dashboard.

### 2.4. Ghi chú upper-bound (Tab 4)

Không cần calculated field, chỉ cần Text Object cố định trên dashboard:
> ⚠️ Đây là ước lượng **cận trên** (upper-bound) dựa trên model kỳ vọng XGBoost, chưa xác thực bằng đo lường thực địa. 95% CI: 525,000 – 748,000 kWh.

---

## 3. Cấu trúc: 5 Dashboards (tabs), mỗi cái gồm nhiều Sheets

Thuật ngữ Tableau: mỗi biểu đồ = 1 **Sheet**; nhiều Sheet ghép lại = 1 **Dashboard** (đây là "tab" trong brief). 5 tab của brief = 5 Dashboard riêng, để trong cùng 1 Workbook, dùng Tab navigation (Show tabs) để chuyển qua lại.

### Dashboard 1 — Operations Overview
| Sheet | Loại | Field |
|---|---|---|
| KPI: Total Masked Loss | Text/BAN | `SUM(TRUE_ENERGY_LOSS_KWH)` = 639,306 kWh |
| KPI: Total CO₂ Loss | Text/BAN | `SUM(TRUE_CO2_LOSS_KG)` = 449,432 kg |
| KPI: Loss Rate % | Text/BAN | `[Loss Rate %]` |
| KPI: Inverters at fault | Text/BAN | `COUNTD(SOURCE_KEY)` filter `[Inverter Worst Severity] <= 2` |
| Open Alerts | Bar/Text | Đếm `TOTAL_LOSS` vs `PARTIAL_LOSS` (dùng `IS_DAY=True`) |
| Plant 1 vs Plant 2 | Bar chart | `SUM(TRUE_ENERGY_LOSS_KWH)` theo `PLANT_NAME` |
| Timeline lỗi theo ngày | Line/Area | `COUNTD` hoặc `SUM(Is Fault)` theo `DATE` (trục X), màu theo `ANOMALY_CLASS` |

Action: click vào bar Plant → Filter action sang Dashboard 2 (Alert Queue), truyền `PLANT_NAME`.

### Dashboard 2 — Alert Queue
Sheet chính: **Table** (crosstab), 1 dòng/inverter, sort giảm dần theo `Inverter Total Loss (kWh)`:
- Columns: `SOURCE_KEY`, `PLANT_NAME`, `Inverter Worst Severity` (hiển thị dạng badge màu), `Inverter Total Loss (kWh)`, `SUM(TRUE_CO2_LOSS_KG)` (fixed theo SOURCE_KEY), `Inverter Recommended Action`, `Inverter Fault Record Count`
- Filter: Plant, khoảng thời gian, dropdown `ANOMALY_CLASS`
- Sort mặc định: `Inverter Total Loss (kWh)` giảm dần

Action: click 1 dòng → Filter action truyền `SOURCE_KEY` sang Dashboard 3, kèm nút "Go to Inverter Detail →" (dashboard navigation button).

### Dashboard 3 — Inverter Detail (drill-down)
- Parameter/Filter chọn `SOURCE_KEY` (nhận từ action của Dashboard 2, hoặc chọn tay qua Filter control)
- Sheet chính: Dual-axis line chart — `AC_POWER` (thực tế) vs `EXPECTED_AC_POWER` (kỳ vọng) theo `DATE_TIME`
- Đánh dấu điểm lỗi: thêm `ANOMALY_CLASS` vào Color trên chính line "Actual" (hoặc thêm 1 Sheet Scatter riêng chỉ hiện điểm lỗi, size lớn hơn) — đỏ = TOTAL_LOSS, cam = PARTIAL_LOSS, ẩn/xám = NORMAL
- Sheet phụ: `POWER_LOSS_KW` theo thời gian (area chart, chỉ ngày)
- KPI nhỏ: tổng tổn thất inverter này, số fault records, khoảng thời gian downtime dài nhất (có thể cần tính thủ công nếu muốn chính xác — xem ghi chú dưới)
- Case study mặc định: set Parameter default = `bvBOhCH3iADSZry`

**Ghi chú kỹ thuật**: "thời gian downtime" (độ dài liên tục của 1 sự cố) đòi hỏi phát hiện chuỗi liên tiếp (consecutive run detection), Tableau làm được bằng table calc `LOD` + `LOOKUP`/`RUNNING_SUM` nhưng khá phức tạp cho người mới dùng cơ bản. Đề xuất: bỏ qua ở v1, chỉ hiển thị timeline trực quan (đã đủ để "thấy" downtime bằng mắt) và số fault records. Nếu sau muốn thêm, báo lại — sẽ hướng dẫn riêng phần LOD run-detection.

### Dashboard 4 — Loss & Carbon Impact
- Breakdown: `SUM(TRUE_ENERGY_LOSS_KWH)` theo Plant × ANOMALY_CLASS (bar chart, stacked hoặc grouped) — khớp bảng Mục 3 phần "Tổn thất theo Plant × Class"
- CO₂ tương tự, dùng `TRUE_CO2_LOSS_KG`
- Top N inverter tệ nhất: Bar chart ngang, sort theo `Inverter Total Loss (kWh)`, dùng **Top N Filter** (Parameter `Top N` int, mặc định 10) trên `SOURCE_KEY`
- Text object cảnh báo upper-bound (mục 2.4 ở trên) — đặt cố định, dễ thấy, không bị che bởi filter

### Dashboard 5 — Plant Comparison (tuỳ chọn)
- Side-by-side: tỷ lệ lỗi (%), phân bố inverter theo class, Loss Rate % — Plant 1 vs Plant 2
- 1 chart duy nhất thể hiện rõ "Plant 2 chiếm >98% tổng lỗi" (ví dụ: 100% stacked bar theo Plant, color = ANOMALY_CLASS)

---

## 4. Global filters & actions cần thiết lập ở Workbook

1. **Date Range filter** (trên `DATE_TIME`) — Apply to Worksheets: **All Using This Data Source** → hiện trên mọi tab, đồng bộ.
2. **Plant filter** — tương tự, áp dụng toàn bộ.
3. **IS_DAY filter** mặc định `True`, nhưng để dạng Quick Filter có thể tắt đi (cho phép xem cả ban đêm nếu người xem tò mò tại sao NORMAL count khác nhau — đúng tinh thần "cả hai" đối tượng: học thuật muốn thấy sự khác biệt, vận hành thì mặc định lọc ngày).
4. Dashboard Actions:
   - Dashboard 1 → Dashboard 2: Filter Action (source: bar Plant, target field `PLANT_NAME`)
   - Dashboard 2 → Dashboard 3: Filter Action (source: table row, target field `SOURCE_KEY`) + nút Navigation

---

## 5. Checklist đối chiếu số liệu trước khi publish

- [ ] KPI Tổng Masked Loss = 639,306.10 kWh
- [ ] KPI Tổng CO₂ = 449,432.19 kg
- [ ] Loss Rate Plant 1 = 0.31%, Plant 2 = 15.25%
- [ ] Bảng Plant × Class khớp đúng 4 số ở Mục 3 brief (2,585.45 / 13,866.68 / 41,294.28 / 581,559.68)
- [ ] Case study `bvBOhCH3iADSZry` hiện đúng inverter Plant 1, thấy rõ 2 khoảng TOTAL_LOSS quanh 07-08/06 và 14/06
- [ ] Không nơi nào trong dashboard tự tính `EXPECTED_AC_POWER - AC_POWER` rồi SUM trực tiếp (sẽ ra raw loss 847k sai) — luôn dùng `TRUE_ENERGY_LOSS_KWH`
- [ ] Text cảnh báo upper-bound hiển thị rõ trên Tab 4
- [ ] Publish lên Tableau Public → kiểm tra data có bị public/download được (đúng như kỳ vọng, vì dataset gốc là Kaggle public, không nhạy cảm)

---

## 6. Việc tiếp theo

Bắt đầu từ Dashboard 1 (Operations Overview) — dễ nhất, dùng để verify KPI trước khi build các tab phức tạp hơn. Báo lại khi làm xong Dashboard 1 để review trước khi qua Dashboard 2 (Alert Queue, có LOD phức tạp hơn).
