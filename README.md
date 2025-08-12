# Fruit Quality Classification

Dự án phân loại độ tươi và loại trái cây bằng ảnh đầu vào. Ảnh đã được xử lý trước (resize, chuẩn hóa) và lưu thành `.npy` để tối ưu tốc độ load khi huấn luyện.

---

## Dataset

Dataset đã được convert thành NumPy arrays (`.npy`) gồm:

- `X.npy`: ảnh đầu vào (đã resize(128, 128)))
- `y_fruit.npy`: nhãn loại trái cây (ví dụ: `orange`, `apple`,...)
- `y_freshness.npy`: nhãn độ tươi (`fresh`, `rotten`)
- `id2fruit_label.json`: 1 dictionary chứa key là id được mã hóa tương ứng với value là tên trái cây
- `freshness_id2label.json`: 1 dictionary chứa key là id được mã hóa tương ứng với value là mức độ của trái cây

📁 Dataset lưu tại:  
[🔗 Google Drive Ảnh](https://drive.google.com/drive/folders/1RzHeizofJqLSi4i-M5FX7USWiNiC3EKh?usp=drive_link)
[🔗 Google Drive Numpy](https://drive.google.com/drive/folders/1NADy3RRIFPnQZLmsVBsf_Q6QV_mPqe51?usp=sharing)

---

## Sử dụng trong Google Colab

```python
!pip install -q gdown

!gdown --folder --remaining-ok https://drive.google.com/drive/folders/1NADy3RRIFPnQZLmsVBsf_Q6QV_mPqe51

import numpy as np
from collections import Counter
import json

X = np.load("Fresh_Rotten_Fruit_Dataset/X.npy")
y_fruit = np.load("Fresh_Rotten_Fruit_Dataset/y_fruit.npy")
y_freshness = np.load("Fresh_Rotten_Fruit_Dataset/y_freshness.npy")

with open('/content/Fresh_Rotten_Fruit_Dataset/id2fruit_label.json', 'r') as f:
    id2fruit_label = json.load(f)
with open('/content/Fresh_Rotten_Fruit_Dataset/freshness_id2label.json', 'r') as f:
    freshness_id2label = json.load(f)

id2fruit_label = {int(k): v for k, v in id2fruit_label.items()}
freshness_id2label = {int(k): v for k, v in freshness_id2label.items()}

fruit_counts = Counter(y_fruit)
fruit_names = [id2fruit_label[i] for i in sorted(fruit_counts.keys())]
fruit_values = [fruit_counts[i] for i in sorted(fruit_counts.keys())]

