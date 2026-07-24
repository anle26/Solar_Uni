# Gói sửa bài main.tex theo phản hồi reviewer (mức sâu — tái định vị contribution)

Sửa `docs/paper/main.tex`. Đây là gói lớn — làm **TỪNG NHÓM theo thứ tự**, sau mỗi nhóm kiểm tra cú pháp LaTeX (hoặc build thử) và báo cáo, **chờ duyệt rồi mới sang nhóm kế**. KHÔNG làm hết một lượt.

## Nguyên tắc chung
- Mọi con số đưa vào phải trích từ code/dữ liệu thật, không bịa.
- Sau mỗi nhóm, quét toàn bài tìm câu cũ mâu thuẫn với nội dung mới — báo cáo nếu tìm thấy.
- Trước khi bắt đầu: `git branch backup-before-review-revision`.
- Không quote nguyên văn >15 từ liên tục từ bài Nahar et al. (chỉ paraphrase — các đoạn dưới đã paraphrase sẵn, giữ nguyên).

---

## NHÓM 1 — Tiêu đề + Tác giả + Abstract

### 1a. Tiêu đề (bỏ "Early-Stage")
Đổi thành:
> An Explainable, Data-Efficient Framework for Multi-Class Fault Diagnosis and Carbon Loss Quantification in Solar PV Monitoring

### 1b. Author block (thay placeholder garbled hiện tại)
```latex
\author{%
\begin{tabular}{c}
Linh Hoang Nguyen\textsuperscript{1},
Tien Dong Dinh\textsuperscript{1,*},
An Dang Thai Le\textsuperscript{1},
Phuc Nguyen Huu Do\textsuperscript{1} \\[0.35em]
\textsuperscript{1}FPT University, Ho Chi Minh Campus, Vietnam \\[0.25em]
\small \texttt{LinhNH67@fe.edu.vn}; \texttt{tienddse200934@fpt.edu.vn}; \texttt{anldtse200518@fpt.edu.vn}; \texttt{do.huuphuc2k5@gmail.com} \\[0.20em]
\end{tabular}%
}
```

### 1c. Abstract (thay toàn bộ)
> Solar photovoltaic (PV) monitoring faces a fundamental tension: the per-inverter local models that achieve the highest forecasting accuracy inadvertently mask the very faults they should reveal, learning the degraded state of an underperforming inverter as its new normal. Moreover, existing approaches often frame fault diagnosis as a binary problem, ignoring the specific operational and maintenance (O\&M) actions required. This paper proposes an explainable, data-efficient framework for multi-class fault diagnosis and carbon loss quantification that inverts the conventional design objective: rather than fitting each inverter as closely as possible, we train a single global Expected Power Model, using Extreme Gradient Boosting, whose deviations expose degraded assets against an ideal plant-wide baseline. Combined with a physics-informed rule-based taxonomy, this design overcomes the anomaly-masking problem inherent to local architectures. We further introduce a Masked Loss Aggregation technique to eliminate Positive Drift Bias in energy loss quantification, enabling faults to be mapped to carbon-emission losses. Evaluated through physics-informed synthetic fault injection on a short (34-day) public benchmark, the framework achieves an F1 score of 0.74 $\pm$ 0.05, and identifies over 639 MWh of recoverable energy loss corresponding to nearly 450 metric tons of avoided CO\textsubscript{2}. The workflow demonstrates that practical inverter-level performance monitoring can be established from short-term operational data alone, without the multi-month or multi-year records required by many long-term degradation studies.

---

## NHÓM 2 — Introduction (thay framing cold-start)

### 2a. Thay đoạn cold-start (đoạn bắt đầu "Solar photovoltaic plants often suffer from an early stage monitoring cold start problem...") bằng 2 đoạn:

**Đoạn 1:**
> Machine learning has been widely applied to PV power forecasting, where the objective is to predict generation as accurately as possible. Achieving this typically favors per-inverter local models that fit each unit's historical behavior closely; for instance, Nahar et al.~\cite{nahar2025} report high forecasting accuracy by training an individual model for every inverter. However, an objective that is ideal for forecasting is counterproductive for fault diagnosis. A model that fits a chronically underperforming inverter closely will learn its degraded output as the expected norm, producing near-zero residuals despite real power loss. We formally define this phenomenon as \textit{anomaly masking}: when an underperforming asset's historical data $X_{\text{local}}$ trains a model $f(X_{\text{local}})$ that normalizes the fault signature, yielding $y - f(X_{\text{local}}) \approx 0$ despite actual degradation. Traditional anomaly detection compounds this by framing diagnosis as a binary normal-versus-anomalous decision, providing limited utility to operations and maintenance (O\&M) teams who require actionable, multi-class categorizations to dispatch crews effectively.

**Đoạn 2:**
> This paper is built on a single observation: the design criterion for an expected-power model should be inverted when the goal shifts from forecasting to fault diagnosis. Rather than minimizing per-inverter error, we deliberately train one global Expected Power Model across an entire plant, so that a degraded inverter is measured against an ideal plant-wide baseline rather than against its own compromised history. The resulting deviations, which reduce forecasting accuracy on degraded plants, are precisely the signal that fault diagnosis requires. Building on this principle, our contributions are: (i) a global expected-power modeling approach that structurally avoids anomaly masking; (ii) a physics-informed, rule-based taxonomy that maps deviations to multi-class, action-oriented fault categories; (iii) a Masked Loss Aggregation technique that eliminates Positive Drift Bias when converting detected faults into energy and carbon-emission losses; and (iv) a demonstration that this workflow operates on short-term operational data alone. Using a 34-day public benchmark, we show that practical inverter-level performance monitoring can be established without the multi-month or multi-year records required by many long-term degradation studies, though we note this necessarily precludes the study of long-horizon phenomena such as seasonal effects and soiling cycles.

### 2b. Rút gọn Related Work để tránh lặp
Đoạn Related Work hiện có (đoạn nhắc Nahar + anomaly masking + "In contrast to localized approaches, our framework utilizes a global model architecture...") giờ TRÙNG với Introduction mới. Rút gọn: BỎ phần định nghĩa anomaly masking và global-vs-local (đã chuyển lên Intro), CHỈ GIỮ phần nói về UML baselines (One-Class SVM, LOF) và XAI (SHAP, XGBoost) cùng các citation tương ứng. Báo cáo đoạn đã rút gọn để tôi xem trước khi giữ.

---

## NHÓM 3 — Đoạn phản biện R² Plant 2 (Results, Section 5.1)

Thay câu hiện tại về Plant 2 ("the lower R² (0.5386) is an intentional and expected artifact... direct performance comparisons with existing localized models (such as Nahar et al.) are purely illustrative...") bằng:

> For Plant 2, the substantially lower R\textsuperscript{2} (0.5386) is not a symptom of global-model misspecification but a direct measurement of the degradation our baseline is designed to expose. This interpretation is supported by three independent observations. First, the accuracy gap is plant-specific: on the well-maintained Plant 1, our global model attains accuracy comparable to the per-inverter local models of Nahar et al.~\cite{nahar2025} (0.9619 vs.\ 0.98), whereas a large divergence appears only on Plant 2 (0.5386 vs.\ 0.91). A systematic bias inherent to the global architecture would degrade both plants comparably; its confinement to Plant 2 instead points to a genuine site-level effect. Second, this localization is corroborated by Nahar et al., who, using an entirely different (localized-forecasting) methodology on the same plants, independently report that Plant 2 exhibits a bimodal inverter-performance distribution with a subset of inverters consistently underperforming (R\textsuperscript{2} below 80\%), which they attribute to localized shading, persistent soiling, or hardware degradation. The chronic underperformance our global baseline surfaces is thus consistent with a physically documented site condition rather than an artifact of our model. Third, the near-identical Train and Test R\textsuperscript{2} (0.5403 vs.\ 0.5386) rules out overfitting as the source of this variance.

---

## NHÓM 4 — Thêm Limitations (circularity + phantom loss)

Thêm vào phần Discussion/Limitations (tạo mục "Limitations" nếu chưa có):

**Đoạn circularity:**
> Two limitations of our evaluation must be stated explicitly. First, the synthetic fault evaluation exhibits partial circularity: the total-loss detection rule (AC power near zero under non-trivial irradiance) closely mirrors the mechanism by which total-loss faults are injected. Consequently, the perfect precision reported in Table~\ref{tab:performance_comparison} should be interpreted as a consistency check confirming that the rule set correctly recovers faults conforming to its own physical assumptions, rather than as evidence of detection capability on independent, real-world faults. Second, no field-verified fault records (e.g., O\&M logs or repair reports) were available to validate detections against ground truth; all reported detection metrics are computed on synthetically injected faults. Validation on real annotated faults remains important future work.

**Đoạn phantom loss:**
> The energy and carbon loss estimates (639 MWh, 449 tCO\textsubscript{2}) depend on the global expected-power baseline, which does not distinguish genuine underperformance from legitimate inter-inverter variability (e.g., differences in siting, capacity, or connection). A portion of the estimated Plant 2 loss (15.25\%) may therefore reflect systematic model bias rather than recoverable loss. While Masked Loss Aggregation removes the random accumulation of positive prediction noise, it does not correct for systematic baseline bias. These figures should thus be read as an upper-bound estimate of recoverable loss, pending validation against actual recovered-energy or repair records.

---

## NHÓM 5 — Quét sạch dấu vết "early-stage/cold-start" còn sót

`grep -ni "early stage\|early-stage\|cold start\|cold-start" main.tex` — với MỌI chỗ còn lại (Conclusion, caption, bất kỳ đâu), sửa cho khớp framing mới (data-efficient / short public benchmark). Dán kết quả grep trước và sau khi sửa.

Đặc biệt kiểm tra Conclusion — nếu có "early stage solar PV monitoring" thì đổi thành "solar PV monitoring" hoặc "short-horizon PV monitoring".

---

## NHÓM 6 — Quick wins

- **(a)** Author block đã xử lý ở Nhóm 1b.
- **(b)** Thêm 1 câu gần Table 2 / Figure 5: giải thích Table 2 tính trên toàn bộ dataset (bao gồm cả các interval ban đêm), trong khi Figure 5 chỉ hiển thị daytime — do đó số NORMAL khác nhau (132,282 full-set vs 72,904 daytime). Đây là nhất quán, không phải lỗi.
- **(c)** Thêm 1 câu ở Section 5.3 (gần Table 3): Cohen's $\kappa$ gần bằng F1 ở mọi phương pháp là do class imbalance mạnh trong tập đánh giá synthetic (tỷ lệ fault thấp), khiến hai chỉ số hội tụ về giá trị tương tự — không phải lỗi tính toán.
- **(d)** Sửa tiêu đề nhúng trong hình: trong `figures/generate_all_figures.py`, các matplotlib title như "Fig. 6", "Fig. 8 — Baseline Comparison" lệch với số Figure thật trong LaTeX (gây rối cho người đọc — reviewer đã nhận xét). Hoặc BỎ HẲN tiêu đề nhúng (để caption LaTeX làm nhiệm vụ đánh số), hoặc sửa cho khớp số Figure LaTeX thật. Ưu tiên BỎ tiêu đề nhúng cho sạch. Regenerate lại các hình bị ảnh hưởng, upload cho người dùng xem.

---

## NHÓM 7 — Bảng ngưỡng taxonomy (Section 3.3)

### 7a. Thêm bảng:
```latex
\begin{table}[h]
\centering
\caption{Rule-Based Fault Taxonomy: Detection Thresholds and O\&M Actions}
\label{tab:taxonomy_thresholds}
\begin{tabular}{lll}
\toprule
\textbf{Class} & \textbf{Detection Condition} & \textbf{O\&M Action} \\
\midrule
TOTAL\_LOSS & $P_{AC}=0$ and $G>0.2$ & Immediate dispatch \\
PARTIAL\_LOSS & $0 < P_{AC} < 0.5\,\bar{P}_{\text{peer}}$ and $G>0.2$ & Inspect within 24h \\
NORMAL & otherwise & None \\
\bottomrule
\end{tabular}
\end{table}
```
Kèm ngay dưới bảng:
> \noindent where $P_{AC}$ is the measured AC power, $G$ is irradiance (kW/m\textsuperscript{2}), and $\bar{P}_{\text{peer}}$ is the mean AC power of co-located inverters at the same timestamp.

### 7b. Câu giải thích BẮT BUỘC (đặt sau bảng — không được bỏ):
> Two design points warrant explanation. First, the PARTIAL\_LOSS rule compares an inverter against the contemporaneous mean of its co-located peers ($\bar{P}_{\text{peer}}$), a relative criterion that complements the absolute plant-wide baseline provided by the global Expected Power Model; the two operate at different granularities and serve as independent corroborating signals rather than a single global comparison. Second, we acknowledge that the TOTAL\_LOSS condition ($P_{AC}=0$ under non-trivial irradiance) closely parallels the mechanism by which total-loss faults are synthetically injected in our evaluation. This overlap is the source of the perfect precision reported in Section~5.3 and is discussed as a limitation there; the thresholds themselves are grounded in physical operating logic (an inverter producing zero power under sunlight is unambiguously faulted) rather than tuned to the injection procedure.

**QUAN TRỌNG:** trước khi thêm bảng, VERIFY lại ngưỡng thật trong `src/taxonomy.py` một lần nữa (đọc code, dán ra) để chắc chắn bảng khớp chính xác code — nếu ngưỡng trong code khác với bảng trên, báo cáo NGAY, không tự sửa.

---

## Sau khi hoàn tất cả 7 nhóm
- Build lại PDF đầy đủ, xác nhận không lỗi cú pháp, không mất reference.
- Quét lần cuối toàn bài đảm bảo nhất quán: không còn "early-stage/cold-start", không còn over-claim "perfect precision" ở Abstract, mọi số khớp Table.
- Báo cáo + gửi PDF để người dùng review.
- KHÔNG commit/push lên GitHub cho tới khi người dùng xác nhận bản cuối.
