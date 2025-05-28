# Ứng dụng Phân loại Nhiễu Ảnh

Ứng dụng này là một công cụ phân loại nhiễu ảnh sử dụng học máy để nhận diện và phân loại các loại nhiễu khác nhau trong ảnh. Ứng dụng sử dụng mô hình Random Forest Classifier để phân loại ảnh thành 4 loại: ảnh sạch, nhiễu Gaussian, nhiễu muối tiêu, và nhiễu đốm.

## Cài đặt

### Yêu cầu hệ thống
- Python 3.6 trở lên
- Các thư viện Python được liệt kê trong file `requirements.txt`

### Cài đặt các thư viện cần thiết
Để cài đặt các thư viện cần thiết, hãy chạy lệnh sau trong terminal:

```bash
pip install -r requirements.txt
```

## Cách chạy ứng dụng

Để khởi động ứng dụng, hãy chạy lệnh sau trong terminal:

```bash
python main.py
```

Sau khi chạy lệnh trên, giao diện đồ họa của ứng dụng sẽ hiển thị.

## Hướng dẫn sử dụng

### 1. Nhập ảnh

- Nhấn nút "Import Image" để chọn và nhập một ảnh từ máy tính của bạn.
- Ảnh sẽ được hiển thị trong khung ảnh ở phía bên trái giao diện.

### 2. Thêm nhiễu vào ảnh (tùy chọn)

Bạn có thể thêm nhiễu vào ảnh để kiểm tra khả năng phân loại của mô hình:

- Nhấn nút "Gaussian Noise" để thêm nhiễu Gaussian vào ảnh.
- Nhấn nút "Salt & Pepper" để thêm nhiễu muối tiêu vào ảnh.
- Nhấn nút "Speckle Noise" để thêm nhiễu đốm vào ảnh.

### 3. Tạo dữ liệu huấn luyện

Trước khi huấn luyện mô hình, bạn cần tạo dữ liệu huấn luyện:

1. Nhấn nút "Generate Training Data".
2. Chọn thư mục chứa các ảnh sạch (không có nhiễu) khi được yêu cầu.
3. Chọn thư mục đầu ra để lưu dữ liệu huấn luyện khi được yêu cầu.
4. Ứng dụng sẽ tự động tạo các phiên bản có nhiễu của các ảnh sạch và lưu chúng vào thư mục đầu ra.

### 4. Huấn luyện mô hình

Sau khi đã tạo dữ liệu huấn luyện, bạn có thể huấn luyện mô hình:

1. Nhập đường dẫn đến thư mục chứa dữ liệu huấn luyện vào ô "Training Data Directory" hoặc nhấn nút "Browse" để chọn thư mục.
2. Nhấn nút "Train Model" để bắt đầu quá trình huấn luyện.
3. Đợi cho đến khi quá trình huấn luyện hoàn tất. Kết quả huấn luyện sẽ được hiển thị trong khung kết quả ở phía bên phải giao diện.

### 5. Lưu mô hình đã huấn luyện

Sau khi huấn luyện mô hình, bạn nên lưu mô hình để sử dụng lại sau này:

1. Nhấn nút "Save Model".
2. Chọn vị trí và tên file để lưu mô hình (định dạng .pkl).

### 6. Tải mô hình đã huấn luyện

Nếu bạn đã có sẵn một mô hình đã huấn luyện, bạn có thể tải nó:

1. Nhấn nút "Load Model".
2. Chọn file mô hình (.pkl) khi được yêu cầu.

### 7. Phân loại ảnh

Sau khi đã huấn luyện hoặc tải mô hình, bạn có thể phân loại ảnh:

1. Nhập ảnh cần phân loại (như đã hướng dẫn ở bước 1).
2. Nhấn nút "Classify Image".
3. Kết quả phân loại sẽ được hiển thị trong khung kết quả ở phía bên phải giao diện, bao gồm:
   - Loại nhiễu được phát hiện
   - Xác suất cho mỗi loại nhiễu (dạng văn bản và biểu đồ)

## Ví dụ sử dụng

### Ví dụ 1: Phân loại ảnh có sẵn

1. Khởi động ứng dụng: `python main.py`
2. Nhấn "Load Model" và chọn một mô hình đã huấn luyện (nếu có)
3. Nhấn "Import Image" và chọn ảnh cần phân loại
4. Nhấn "Classify Image" để xem kết quả phân loại

### Ví dụ 2: Huấn luyện mô hình mới

1. Khởi động ứng dụng: `python main.py`
2. Nhấn "Generate Training Data"
3. Chọn thư mục chứa ảnh sạch
4. Chọn thư mục đầu ra cho dữ liệu huấn luyện
5. Nhập đường dẫn đến thư mục dữ liệu huấn luyện vào ô "Training Data Directory"
6. Nhấn "Train Model" để huấn luyện mô hình
7. Nhấn "Save Model" để lưu mô hình đã huấn luyện

### Ví dụ 3: Kiểm tra hiệu suất mô hình với nhiễu nhân tạo

1. Khởi động ứng dụng: `python main.py`
2. Nhấn "Load Model" và chọn một mô hình đã huấn luyện
3. Nhấn "Import Image" và chọn một ảnh sạch
4. Nhấn một trong các nút thêm nhiễu (ví dụ: "Gaussian Noise")
5. Nhấn "Classify Image" để xem liệu mô hình có phân loại đúng loại nhiễu không

## Lưu ý

- Quá trình tạo dữ liệu huấn luyện và huấn luyện mô hình có thể mất một khoảng thời gian tùy thuộc vào số lượng ảnh và cấu hình máy tính của bạn.
- Để có kết quả phân loại tốt nhất, nên sử dụng một tập dữ liệu huấn luyện đa dạng với nhiều loại ảnh khác nhau.
- Mô hình đã huấn luyện được lưu dưới định dạng .pkl và có thể được tải lại để sử dụng trong các phiên làm việc sau.