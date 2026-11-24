# Nguồn và phạm vi phụ lục

| Phần | Đầu vào | Cách tái lập |
| --- | --- | --- |
| Figure 6–7 | Hai CSV Git LFS lịch sử tại `5b2d812` | Tính lũy kế thời gian, chọn bằng validation trong từng bin và tổng hợp theo metric của script gốc. Kiểm tra 624 hàng với R. |
| Figure 8–11 | Bản 10K và 50K trong cùng hai CSV lịch sử | Giữ cùng nhóm 4, 3, 2 và 5 bộ dữ liệu. Giữ riêng mức phân vị của từng nhánh theo script. |
| Figure 12–17 | CSV Figshare bản 10K và lịch sử bản 50K | Vẽ điểm test thô từng bộ dữ liệu, chặn R² âm ở 0 như script. |
| Figure 18, 20–24 | `random_search_xps.csv` | Xáo các lượt công bố rồi lấy test của cấu hình có validation tốt nhất. Figure 21 giữ dải theo 15 hoán vị của script và 30 của chú thích. |
| Figure 33–34 | Bốn CSV Bayesian Git LFS và CSV numerical lịch sử | Giữ thứ tự thời gian của Bayes. Xáo random 15 lần, có default ở hạng 0. |
| Figure 39–40 | Checkpoint `5b2d812` cho nhánh cố định, `7265a3f` cho nhánh tune | Đối chiếu 10.782 hàng với R. 24 đường mặc định được so với tọa độ trong PDF. |
| A.1 · A.3 | PDF phụ lục trang 15–16 và 22–23 | 57 thiết lập đã đối chiếu OpenML; 70 mục trong 8 bảng HPO. |
| A.4 | CSV ablation và bốn tệp `tests_*.csv` công bố | Chạy lại bốn hồi quy, đối chiếu 34 hệ số với R. Cột tác giả giữ số đã làm tròn từ CSV lịch sử. |

Dấu chấm ở hạng mặc định dùng dữ liệu công bố. Hình trong notebook vẽ lại bằng Python và cố định seed 0 cho hoán vị; dữ liệu 50K là kết quả đã công bố của tác giả.

Bộ nộp có 40 notebook figure đã chạy. Trong số đó, 27 hình tái lập phân tích từ kết quả công bố.

| Hình số hóa | Dữ liệu lấy trực tiếp từ PDF | Giới hạn |
| --- | --- | --- |
| 19 | Hai bảng màu cắt theo vùng phóng trên Electricity. Mỗi ô trong lưới 260 × 240 được phân nhóm theo chênh lệch kênh đỏ và xanh. | Trục là vị trí trong vùng phóng. Chưa có xác suất dự đoán hoặc checkpoint RF/MLP của hình. |
| 25–32 | Nhãn, tọa độ nút và hai đường Bézier của từng nhánh. | Độ rộng dùng để tính tỷ lệ trong sơ đồ. Chưa có metadata để xác nhận số đếm. Figure 28 có chênh lệch khoảng 1,13% giữa đầu và cuối, được giữ nguyên. |
| 35–38 | Đường trung bình và dải vector theo trục log. | Các tọa độ có giới hạn độ chính xác của PDF. Ba mốc CSV lịch sử đã ghim hash không chứa lượt được đánh dấu 40K ở cột trần số vòng hoặc số cây; chưa tìm được đầu vào chính xác cho nhánh này. |

Trong Figure 39–40, ba mô hình ở cả hai nhánh dùng đúng các nhóm 15, 19, 7 và 13 bộ dữ liệu. Hồi quy số loại isolet theo script của tác giả. Hạng 0 kiểm tra nguồn vì điểm này không phụ thuộc vào thứ tự hoán vị. Các đường cong chạy lại dùng seed 0.

Chạy `python -m src.verify_appendix --reanalyze` để đối chiếu lại dữ liệu với R. `src/reference/` có các wrapper để chạy lại bản R từ projection trong `.cache/`. Dùng thư mục `analyses` của repo tác giả tại commit `9d54cf5` và R có `tidyverse`, `zoo`, `RColorBrewer`. Ví dụ `Rscript src/reference/time.R 6 <author-analyses> .cache figure_06_r.csv`. Wrapper A.4 dùng số 2–5 tương ứng bốn thí nghiệm; Figure 21 theo chú thích dùng số 121.

[Đối chiếu Python–R](appendix_parity.json) · [Kiểm tra độc lập CSV lịch sử](appendix_input_verification.json) · [Kiểm chứng số hóa và Figure 39–40](published_parity.json) · [Rà soát cấu hình số cây](published_input_audit.json)

[Phụ lục NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Supplemental-Datasets_and_Benchmarks.pdf) · [Mã và CSV lịch sử](https://github.com/LeoGrin/tabular-benchmark/tree/5b2d812329d191f949f9d085616a6c41d3198688/analyses)
