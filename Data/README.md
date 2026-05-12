Repository Data bao gồm 03 thành phần chính phục vụ quá trình khai thác dữ liệu:

### 1. Google Colab Notebook (đuôi .ipynb)
* **Mục đích:** Phục vụ làm việc nhóm, cộng tác trực tuyến và chạy thử nghiệm nhanh (Prototyping).
* **Đặc điểm:** * Sử dụng thư viện `vnstock` kết nối qua API của các công ty chứng khoán (mặc định KBS).
    * **Giới hạn:** Do đặc thù môi trường Cloud và chính sách API, script này chỉ hỗ trợ crawl dữ liệu của 02 hợp đồng tương lai gần nhất (hợp đồng hiện tại và hợp đồng kế tiếp).
    * Phù hợp để cập nhật dữ liệu Real-time hoặc Intraday ngắn hạn.

### 2. Local Python Script (đuôi .py)
* **Mục đích:** Khai thác dữ liệu lịch sử chuyên sâu (Deep History Data).
* **Đặc điểm:** * Chạy trên môi trường Local (VS Code/PyCharm).
    * Sử dụng thư viện VCI, có thể truy xuất dữ liệu quá khứ của các hợp đồng đã đáo hạn, phục vụ việc huấn luyện mô hình Machine Learning cần độ sâu thời gian lớn.

### 3. Raw Dataset (đuôi .csv)
* **Mô tả:** Tập dữ liệu thô đã được trích xuất thành công từ script Local, gồm 2,953 dòng dữ liệu từ ngày 11/09/2023 đến ngày 12/05/2026
* **Các trường dữ liệu:** * `time`: Thời gian giao dịch.
    * `open`, `high`, `low`, `volume`: Các thông số kỹ thuật cơ bản.
    * `vn30_index`: Chỉ số cơ sở VN30 (Biến độc lập).
    * `y_f1m`: Giá đóng cửa VN30F1M (Biến mục tiêu cho bài toán hồi quy).

## Cách cài đặt & Sử dụng

### Yêu cầu hệ thống
* Python 3.9+
* Thư viện: `vnstock`, `pandas`, `openpyxl`

### Cài đặt môi trường
```bash
pip install vnstock -U
pip install pandas
