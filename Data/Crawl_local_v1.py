import pandas as pd
from vnstock.api.quote import Quote
from datetime import datetime
import os

def collect_derivative_data():
    print("--- Bắt đầu crawl data trên môi trường Local ---")

    # 1. Cấu hình thời gian và đường dẫn
    START_DATE = '2020-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Lưu file tại thư mục chứa code
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'vn30f1m_raw_dataset.csv')

    try:
        # 2. Crawl data VN30F1M
        print(f"1. Đang lấy dữ liệu lịch sử VN30F1M từ {START_DATE}...")
        q_f1m = Quote(symbol='VN30F1M', source='VCI')
        df_f1m = q_f1m.history(start=START_DATE, end=END_DATE, interval="1h")

        # 3. Crawl data VN30
        print("2. Đang lấy dữ liệu chỉ số cơ sở VN30...")
        q_vn30 = Quote(symbol='VN30', source='VCI')
        df_vn30 = q_vn30.history(start=START_DATE, end=END_DATE, interval="1h")

        # 4. Xử lý dữ liệu
        if df_f1m.empty or df_vn30.empty:
            print("Cảnh báo: Một trong hai nguồn dữ liệu bị trống!")
            return

        df_f1m = df_f1m[['time', 'open', 'high', 'low', 'close', 'volume']].rename(columns={'close': 'y_f1m'})
        df_vn30 = df_vn30[['time', 'close']].rename(columns={'close': 'vn30_index'})
        
        # Merge dữ liệu
        raw_data = pd.merge(df_f1m, df_vn30, on='time', how='inner')

        # 5. Xuất dữ liệu ra CSV
        raw_data.to_csv(file_path, index=False, encoding='utf-8-sig')

        print(f"--- Hoàn tất! ---")
        print(f"Số dòng: {len(raw_data)}")
        print(f"Lưu tại: {file_path}")
        print(raw_data.head())

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    collect_derivative_data()