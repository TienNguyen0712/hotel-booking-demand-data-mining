![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-purple)

# 🏨 Hotel Booking Demand – Data Mining Project

## 📌 Giới thiệu
Trong bối cảnh ngành khách sạn chịu ảnh hưởng mạnh bởi hành vi đặt phòng và tỷ lệ hủy booking, việc khai phá dữ liệu lịch sử đặt phòng giúp doanh nghiệp:
- Dự đoán rủi ro hủy phòng
- Phân khúc khách hàng
- Phát hiện các mô hình đặt phòng phổ biến
- Phân tích xu hướng theo thời gian

Đề tài này áp dụng **quy trình Khai phá dữ liệu (Data Mining)** để khám phá tri thức tiềm ẩn từ **Hotel Booking Demand dataset**, đáp ứng đầy đủ yêu cầu môn học *Khai phá dữ liệu*.

---

## 🎯 Mục tiêu & Câu hỏi nghiên cứu

### Mục tiêu
- Áp dụng toàn bộ pipeline Khai phá dữ liệu:  
  **Tiền xử lý → Phân tích mô tả → Mô hình hóa → Đánh giá → Insight**
- Thực nghiệm và so sánh nhiều thuật toán khai phá dữ liệu
- Rút ra insight có ý nghĩa cho bài toán kinh doanh khách sạn

### Câu hỏi nghiên cứu
1. Có thể **dự đoán khả năng hủy booking** của khách hàng không?
2. Có thể **phân khúc khách hàng** dựa trên hành vi đặt phòng không?
3. Những **luật kết hợp** nào thường xuất hiện trong dữ liệu booking?
4. Xu hướng **đặt phòng và hủy phòng thay đổi như thế nào theo thời gian**?

---

## 📂 Dataset

- **Tên:** Hotel Booking Demand
- **Nguồn:** Public dataset (Kaggle – dữ liệu nghiên cứu học thuật)
- **Số dòng:** ~119,390
- **Số cột:** 32
- **Đối tượng:** Booking của City Hotel và Resort Hotel

### Một số thuộc tính quan trọng
- `is_canceled`: Trạng thái hủy booking (target)
- `lead_time`: Số ngày từ lúc đặt đến ngày nhận phòng
- `adr`: Giá trung bình mỗi ngày
- `arrival_date_*`: Thông tin thời gian
- `adults`, `children`, `stays_in_weekend_nights`

---

## 🧪 Quy trình Khai phá dữ liệu

### 1️⃣ Tiền xử lý dữ liệu
- Xử lý missing values (`children`, `agent`, `company`)
- Loại bỏ booking không hợp lệ
- Encode biến categorical
- Chuẩn hóa dữ liệu cho các mô hình cần thiết

### 2️⃣ Phân tích mô tả (EDA)
- Thống kê cơ bản
- Histogram, boxplot, scatter plot
- Heatmap tương quan
- Phân tích tỷ lệ hủy booking

---

## 🤖 Các kỹ thuật Khai phá dữ liệu được sử dụng

### 🔹 Phân lớp (Classification)
**Mục tiêu:** Dự đoán khách có hủy booking hay không  
**Thuật toán:**
- Logistic Regression
- Decision Tree
- Random Forest  

**Đánh giá:**
- Accuracy
- Precision / Recall
- F1-score
- Confusion Matrix

---

### 🔹 Phân cụm (Clustering)
**Mục tiêu:** Phân khúc khách hàng đặt phòng  

**Thuật toán:**
- K-Means
- Hierarchical Clustering  

**Đánh giá:**
- Elbow Method
- Silhouette Score

---

### 🔹 Khai phá Luật kết hợp (Association Rules)
**Mục tiêu:** Phát hiện các mẫu hành vi đặt phòng phổ biến  

**Thuật toán:**
- Apriori  

**Độ đo:**
- Support
- Confidence
- Lift

---

### 🔹 Phân tích Chuỗi thời gian (Time Series)
**Mục tiêu:** Phân tích xu hướng booking theo thời gian  

- Số booking theo tháng
- Tỷ lệ hủy booking theo thời gian
- Moving Average & Decomposition

---

## 📊 Kết quả & Insight chính
- Lead time cao có tương quan mạnh với khả năng hủy booking
- City Hotel có tỷ lệ hủy cao hơn Resort Hotel
- Tồn tại các nhóm khách hàng rõ ràng dựa trên giá và thời gian lưu trú
- Booking có tính mùa vụ theo tháng

---

## 🗂️ Cấu trúc thư mục
```text
hotel-booking-demand-data-mining/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── hotel_bookings.csv
|   ├── interim/
│   │   └── hotel_bookings_cleaned.csv # Chỉ tiền xử lý
│   ├── processed/ 
│   │   ├── hotel_bookings_processed.csv  # Tiền xử lý sinh dặc trưng mới
│   │   └── hotel_bookings_ts.csv # Bộ dữ liệu sử dụng rieng cho chuỗi thời gian
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_clustering.ipynb
│   ├── 06_association_rules.ipynb
│   └── 07_time_series.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── evaluation.py
│   └── utils.py
│
├── reports/
│   ├── report.docx
│   ├── slides.pptx
│   └── figures/
│
├── requirements.txt
└── .gitignore
```

---

## 🚀 Công nghệ & Thư viện
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- mlxtend
- statsmodels

---

## ⚠️ Hạn chế & Hướng mở rộng
- Dataset không phản ánh dữ liệu thời gian thực
- Chưa tối ưu hyperparameter chuyên sâu
- Có thể mở rộng:
  - Ứng dụng Streamlit
  - Dự báo booking (ARIMA/Prophet)
  - Explainable AI (SHAP)

---

## 👨‍🎓 Thông tin học thuật
- Đề tài phục vụ học phần **Khai phá dữ liệu**
- Sản phẩm là **bài làm học thuật gốc**
- Các tài liệu, thư viện được trích dẫn rõ ràng

---

## 📎 Tài liệu tham khảo
- Antonio, N., Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief.
- Kaggle: Hotel Booking Demand Dataset
