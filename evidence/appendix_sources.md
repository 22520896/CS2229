# Nguồn và phạm vi phụ lục

| Phần | Đầu vào | Cách tái lập |
| --- | --- | --- |
| Figure 6–7 | Hai CSV Git LFS lịch sử tại `5b2d812` | Tính lũy kế thời gian, chọn bằng validation trong từng bin và tổng hợp theo metric của script gốc. Kiểm tra 624 hàng với R. |
| Figure 8–11 | Bản 10K và 50K trong cùng hai CSV lịch sử | Giữ cùng nhóm 4, 3, 2 và 5 bộ dữ liệu. Giữ riêng mức phân vị của từng nhánh theo script. |
| Figure 12–17 | CSV Figshare bản 10K và lịch sử bản 50K | Vẽ điểm test thô từng bộ dữ liệu, chặn R² âm ở 0 như script. |
| Figure 18, 20–24 | `random_search_xps.csv` | Xáo các lượt công bố rồi lấy test của cấu hình có validation tốt nhất. Figure 21 giữ dải theo 15 hoán vị của script và 30 của chú thích. |
| Figure 33–34 | Bốn CSV Bayesian Git LFS và CSV numerical lịch sử | Giữ thứ tự thời gian của Bayes. Xáo random 15 lần, có default ở hạng 0. |
| A.1 · A.3 | PDF phụ lục trang 15–16 và 22–23 | 57 thiết lập đã đối chiếu OpenML; 70 mục trong 8 bảng HPO. |
| A.4 | CSV ablation và bốn tệp `tests_*.csv` công bố | Chạy lại bốn hồi quy, đối chiếu 34 hệ số với R. Cột tác giả giữ số đã làm tròn từ CSV lịch sử. |

Dấu chấm ở hạng mặc định dùng dữ liệu công bố. Hình trong notebook vẽ lại bằng Python và cố định seed 0 cho hoán vị; dữ liệu 50K là kết quả đã công bố của tác giả.

Bộ nộp tái lập 25/40 figure. Figure 19 cần lưới dự đoán hoặc đặc tả đầy đủ cho thí nghiệm vẽ đường biên. Figure 25–32 cần bảng đặc trưng dữ liệu tại thời điểm lọc. Figure 35–40 cần đủ đầu vào của các nhánh thay đổi giới hạn, patience và tuning số cây. Các nguồn công khai đã đối chiếu chưa đủ để tái lập các hình này.

Chạy `python -m src.verify_appendix --reanalyze` để đối chiếu lại dữ liệu với R. `src/reference/` có các wrapper để chạy lại bản R từ projection trong `.cache/`. Dùng thư mục `analyses` của repo tác giả tại commit `9d54cf5` và R có `tidyverse`, `zoo`, `RColorBrewer`. Ví dụ `Rscript src/reference/time.R 6 <author-analyses> .cache figure_06_r.csv`. Wrapper A.4 dùng số 2–5 tương ứng bốn thí nghiệm; Figure 21 theo chú thích dùng số 121.

[Đối chiếu Python–R](appendix_parity.json) · [Kiểm tra độc lập CSV lịch sử](appendix_input_verification.json)

[Phụ lục NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Supplemental-Datasets_and_Benchmarks.pdf) · [Mã và CSV lịch sử](https://github.com/LeoGrin/tabular-benchmark/tree/5b2d812329d191f949f9d085616a6c41d3198688/analyses)
