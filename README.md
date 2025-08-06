# Fruit Quality Classification

Dự án phân loại độ tươi và loại trái cây bằng ảnh đầu vào. Ảnh đã được xử lý trước (resize, chuẩn hóa) và lưu thành `.npy` để tối ưu tốc độ load khi huấn luyện.

---

## Dataset

Dataset đã được convert thành NumPy arrays (`.npy`) gồm:

- `X.npy`: ảnh đầu vào (đã resize(128, 128)))
- `y_fruit.npy`: nhãn loại trái cây (ví dụ: `orange`, `apple`,...)
- `y_freshness.npy`: nhãn độ tươi (`fresh`, `rotten`)

📁 Dataset lưu tại:  
[🔗 Google Drive Ảnh](https://drive.google.com/drive/folders/1RzHeizofJqLSi4i-M5FX7USWiNiC3EKh?usp=drive_link)
[🔗 Google Drive Numpy](https://drive.google.com/drive/folders/1NADy3RRIFPnQZLmsVBsf_Q6QV_mPqe51?usp=sharing)

---

## Sử dụng trong Google Colab

```python
!pip install -q gdown

!gdown --folder --remaining-ok https://drive.google.com/drive/folders/1NADy3RRIFPnQZLmsVBsf_Q6QV_mPqe51

# Load dữ liệu numpy
import numpy as np

X = np.load("Fresh_Rotten_Fruit_Dataset/X.npy")
y_fruit = np.load("Fresh_Rotten_Fruit_Dataset/y_fruit.npy")
y_freshness = np.load("Fresh_Rotten_Fruit_Dataset/y_freshness.npy")
