![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-purple)

# Hotel Booking Demand – Data Mining Project

## 1. Giới thiệu
Dự án này sử dụng bộ dữ liệu **Hotel Booking Demand** nhằm áp dụng và phân tích các kỹ thuật trong lĩnh vực **Khai phá dữ liệu (Data Mining)**.  
Các bài toán chính được nghiên cứu trong dự án bao gồm:

- Phân loại (Classification)
- Phân cụm (Clustering)
- Chuỗi thời gian (Time Series)
- Luật kết hợp (Association Rules)

Mục tiêu của dự án là khai thác tri thức từ dữ liệu đặt phòng khách sạn, hỗ trợ việc phân tích hành vi khách hàng và ra quyết định trong lĩnh vực kinh doanh khách sạn.

---

## 2. Dataset
- **Tên dataset**: Hotel Booking Demand  
- **Nguồn**: Kaggle / UCI Machine Learning Repository  
- **Số lượng bản ghi**: Khoảng 119.000  
- **Lĩnh vực**: Khách sạn – Du lịch  

### Các nhóm thuộc tính chính:
- Thông tin đặt phòng: `lead_time`, `adr`, `stays_in_week_nights`, …
- Thông tin khách hàng: `customer_type`, `market_segment`, …
- Thời gian đến: `arrival_date_year`, `arrival_date_month`, …
- Nhãn mục tiêu: `is_canceled`

---

## 3. Cấu trúc repository

```text
hotel-booking-demand-mining/
├─ README.md
├─ requirements.txt
├─ configs/
│ ├─ base.yaml
│ ├─ classification.yaml
│ ├─ clustering.yaml
│ ├─ association.yaml
│ └─ timeseries.yaml
├─ data/
│ ├─ raw/ # dữ liệu gốc
│ ├─ interim/ # dữ liệu trung gian sau tiền xử lý
│ └─ processed/ # dữ liệu cuối cho modeling
├─ notebooks/
│ ├─ 00_eda.ipynb
│ ├─ 10_classification.ipynb
│ ├─ 20_clustering.ipynb
│ ├─ 30_timeseries.ipynb
│ └─ 40_association_rules.ipynb
├─ src/
│ ├─ data/
│ ├─ classification/
│ ├─ clustering/
│ ├─ timeseries/
│ └─ association/
├─ reports/
│ ├─ figures/
│ └─ final_report.md
└─ tests/
```
---

## 4. Tiền xử lý dữ liệu
Dữ liệu được xử lý theo quy trình chuẩn:

### Các bước chính:
- Xử lý giá trị thiếu
- Chuẩn hóa và tạo biến thời gian
- Loại bỏ các thuộc tính không cần thiết
- Tách dataset phù hợp cho từng bài toán

---

## 5. Phân loại (Classification)

### Mục tiêu
Dự đoán khả năng **hủy đặt phòng** (`is_canceled`).

### Thuật toán sử dụng
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)

### Chỉ số đánh giá
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 6. Phân cụm (Clustering)

### Mục tiêu
Phân nhóm các booking/khách hàng dựa trên hành vi đặt phòng.

### Thuật toán
- KMeans
- MiniBatchKMeans

### Chỉ số đánh giá
- Silhouette Score
- Calinski–Harabasz Index

---

## 7. Chuỗi thời gian (Time Series)

### Mục tiêu
Phân tích và dự báo số lượng booking theo thời gian.

### Phương pháp
- SARIMAX (Seasonal ARIMA with eXogenous variables)

### Chỉ số đánh giá
- RMSE
- MAPE

Dữ liệu chuỗi thời gian được tổng hợp theo chu kỳ ngày hoặc tuần.

---

## 8. Luật kết hợp (Association Rules)

### Mục tiêu
Phát hiện các mối quan hệ thường xuyên giữa các thuộc tính trong đặt phòng khách sạn.

### Thuật toán
- Apriori

### Chỉ số đánh giá
- Support
- Confidence
- Lift

Các luật kết hợp được phân tích để rút ra ý nghĩa thực tiễn trong hoạt động kinh doanh khách sạn.

---

## 9. Hướng dẫn chạy dự án

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```
### Chạy notebook
```bash
jupyter lab
```
Thực hiện chạy các notebook trong thư mục `notebooks/` theo đúng thứ tự

---

## 10. Kết luận 

Dự án cho thấy việc áp dụng các kỹ thuật khai phá dữ liệu giúp:
* Hiểu rõ hơn hành vi và xu hướng của khách hàng
* Hỗ trợ dự báo nhu cầu đặt phòng
* Khai thác các tri thức tiềm ẩn từ dữ liệu lớn trong lĩnh vực khách sạn

---

## 11. Tài liệu tham khảo
* Moro et al., Hotel Booking Demand Datasets, Data in Brief
* Han, Kamber & Pei, Data Mining: Concepts and Techniques
* Scikit-learn Documentation
* Statsmodels Documentation

