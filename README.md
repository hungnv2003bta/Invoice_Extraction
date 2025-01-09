# Invoice Extraction Project

## Hướng dẫn Build và Thực thi Code

### 1. Cài đặt Thư viện Cần thiết
Cài đặt các thư viện trong file `requirements.txt` bằng lệnh sau:  
```bash
pip install -r requirements.txt
```

### 2. Cấu hình Đường dẫn
- Điều chỉnh đường dẫn ảnh input và file JSON trong mã nguồn sao cho phù hợp với cấu trúc dữ liệu của bạn.

### 3. Một số File Quan trọng
- **`main/result_with_threshold.json`**: Lưu kết quả xử lý cuối cùng.
- **`main/Invoice_Extraction.ipynb`**: Notebook dùng để thực thi code từng bước, dễ dàng hình dung từng giai đoạn xử lý bài toán.
- **`streamlit/invoice_extraction.py`**: Thực thi code và trả về kết quả xử lý dữ liệu.
- **`streamlit/streamlit.py`**: File dùng để triển khai ứng dụng trên Streamlit.

### 4. Chạy Ứng dụng với Streamlit
Thực thi lệnh sau để chạy ứng dụng Streamlit:
```bash
streamlit run streamlit/streamlit.py
```

## Link Dataset
Dataset sử dụng trong dự án: [MC-OCR Scanned Receipt Dataset](https://www.kaggle.com/datasets/hariwh0/mcocr-scanned-receipt)
