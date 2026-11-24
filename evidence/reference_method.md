# Đối chiếu đầu ra Python với R

Hai tệp tác giả được kiểm SHA-256 bằng giá trị trong `src/author_data.py`. Bản thu gọn giữ thứ tự dòng, các cột nhận diện nhóm, trạng thái default/random, điều kiện thí nghiệm và điểm validation/test/time.

Kho `r_reference.zip` chứa đầu ra từ bước phân tích R độc lập trên dữ liệu tác giả. Cố định seed 0 cho từng phép đối chiếu và giữ chuỗi liên tục giữa hai nhánh Figure 5 như vòng lặp gốc. Xem [biên bản sai số Python–R](python_r_parity.json). Lệnh `python -m src.verify --reanalyze` tính lại rồi kiểm tra khóa, vị trí giá trị thiếu, số dòng và sai số tuyệt đối tới 1e-12.

| Hình | Tham chiếu R tại commit 9d54cf5 | Giá trị đối chiếu |
| --- | --- | --- |
| 1 | `plot_utils.R::plot_aggregated_results` | Điểm trung bình và dải min–max đến hạng 400, bốn nhóm |
| 2 | `high_frequency.R` | Điểm trung bình theo từng hoán vị, mức làm trơn và mô hình |
| 3 | `uninformative_features.R` | Nhánh 30 hoán vị, hạng 20, trung bình và phân vị 10%–90% |
| 4 | `remove_uninformative_features.R` và `add_uninformative_features.R` | Trung bình và dải từng tỷ lệ, lưu cả phương án caption của 4b |
| 5 | `random_rotation.R` | Điểm theo hoán vị, nhóm bỏ 0%/50% và trạng thái xoay |

Figure 3 cố định seed trước nhánh 30 hoán vị dùng cho hình. Figure 4b giữ hai phương án đã ghi trong paper và script công bố. Figure 5 giữ cả hai điều kiện đã có trong dữ liệu, script vẽ chính đặt bộ lọc cho một điều kiện tại một lần xuất.

Các tệp `v2_*.csv` là trích xuất từ kết quả đã kiểm chứng, gồm đường cong A.1, một pilot làm trơn và các cặp phép xoay cấu hình mặc định. Phạm vi và hash đầu vào nằm trong `project_snapshot.json`.

[Mã phân tích tác giả](https://github.com/LeoGrin/tabular-benchmark/tree/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/analyses)
