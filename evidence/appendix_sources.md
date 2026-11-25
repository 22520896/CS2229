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

Bộ nộp báo 27/40 hình tái lập phân tích từ kết quả công bố. Mỗi lượt có điểm validation/test đã lưu. Notebook chạy lại bước chọn cấu hình theo validation, rồi tổng hợp điểm test của các cấu hình được chọn.

## Đầu vào cần để chạy lại 13 hình còn thiếu

| Hình | Đầu vào và thao tác kiểm chứng | Hiện trạng |
| --- | --- | --- |
| 19 | Dùng Electricity, khớp hai đặc trưng, biến đổi Gaussian và split gốc. Train lại RF/MLP theo cấu hình mặc định của thí nghiệm hoặc nạp checkpoint gốc, đánh giá trên split giữ riêng và xuất dự đoán trên lưới. | Có dữ liệu, mô tả và mặc định trong repo. Hai đặc trưng, split và cấu hình chính xác của hình chưa được xác minh đầy đủ. |
| 25–32 | Khôi phục danh sách dữ liệu ứng viên và trạng thái theo từng bước lọc của tác giả. Chạy lại quy tắc và đếm ID dữ liệu ở mỗi nhánh. | Có mô tả quy tắc và mã dựng Sankey. Thiếu metadata ứng viên ban đầu và log quyết định lọc. |
| 35–38 | Dùng đúng cặp bộ dữ liệu, splits, mô hình và cấu hình lấy mẫu. Chạy nhánh giới hạn thấp và 40K, lưu validation/test và thời gian từng lượt. Dựng lại đường cong theo số lượt và theo thời gian. | Paper ghi trần, patience và cấu hình mặc định. Các CSV lịch sử đã kiểm tra chưa có kết quả từng lượt của nhánh 40K. |

Có ba mốc kiểm chứng. Khi có kết quả từng lượt, chạy lại phân tích chọn bằng validation và tổng hợp test. Khi có checkpoint và split gốc, nạp mô hình để đo lại điểm. Khi có mã huấn luyện, cấu hình, phiên bản dữ liệu, split và seed, có thể fit lại rồi đánh giá. Cấu hình được chọn bằng validation trước khi tổng hợp test. Phạm vi hoàn tất thực nghiệm được tính từ lượt chạy có đầu ra kiểm chứng.

Trong Figure 39–40, ba mô hình ở cả hai nhánh dùng đúng các nhóm 15, 19, 7 và 13 bộ dữ liệu. Hồi quy số loại isolet theo script của tác giả. Hạng 0 kiểm tra nguồn vì điểm này không phụ thuộc vào thứ tự hoán vị. Các đường cong chạy lại dùng seed 0.

Chạy `python -m src.verify_appendix --reanalyze` để đối chiếu lại dữ liệu với R. `src/reference/` có các wrapper để chạy lại bản R từ projection trong `.cache/`. Dùng thư mục `analyses` của repo tác giả tại commit `9d54cf5` và R có `tidyverse`, `zoo`, `RColorBrewer`. Ví dụ `Rscript src/reference/time.R 6 <author-analyses> .cache figure_06_r.csv`. Wrapper A.4 dùng số 2–5 tương ứng bốn thí nghiệm; Figure 21 theo chú thích dùng số 121.

[Đối chiếu Python–R](appendix_parity.json) · [Kiểm tra độc lập CSV lịch sử](appendix_input_verification.json) · [Kiểm chứng Figure 39–40](published_parity.json) · [Rà soát cấu hình số cây](published_input_audit.json)

Tư liệu đọc hình đã chuyển ra khỏi cây thư mục bản nộp. [Bản lưu trong lịch sử Git](https://github.com/22520896/CS2229/tree/965690418d842a66190f0bd92abbfdb982ffbe20)

[Phụ lục NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Supplemental-Datasets_and_Benchmarks.pdf) · [Mã và CSV lịch sử](https://github.com/LeoGrin/tabular-benchmark/tree/5b2d812329d191f949f9d085616a6c41d3198688/analyses)
