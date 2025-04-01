# 1. 필수 라이브러리 설치 (터미널에서 실행)
# !pip install yfinance pandas numpy tensorflow scikit-learn matplotlib

# 2. 데이터 수집 및 전처리
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 데이터 다운로드 (3년치)
ticker = "AMZN"
end_date = "2025-03-21"
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=3)  # 3년 전 날짜 자동 계산

data = yf.download(ticker, start=start_date, end=end_date)
data.to_csv("AMZN_3years.csv")  # 데이터 백업 저장

# 데이터 클렌징
data = data[['Close']]  # 종가만 선택
data = data.dropna()  # 결측치 제거

# 정규화 (0~1 사이로 스케일링)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. 시퀀스 생성 (LSTM/GRU 입력 형식)
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60  # 60일 창 크기
X, y = create_sequences(scaled_data, window_size)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (샘플 수, 타임스텝, 피처 수)

# 4. 하이브리드 모델 구축 (LSTM + GRU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Bidirectional, Conv1D

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    Bidirectional(LSTM(100, return_sequences=True)),
    Bidirectional(GRU(50)),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. 모델 훈련 및 검증
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6. 미래 5거래일 예측
def predict_future_days(model, last_sequence, scaler, pred_days=5):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(pred_days):
        # 다음 날 예측
        x_input = current_sequence.reshape(1, window_size, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)
        
        # 시퀀스 업데이트 (첫 번째 값 제거, 새 예측값 추가)
        current_sequence = np.append(current_sequence[1:], pred)
    
    return scaler.inverse_transform([predictions])[0]

# 7. 예측 실행 및 결과 처리
last_date = data.index[-1]
last_sequence = scaled_data[-window_size:]  # 최근 60일 데이터

# 평일만 계산 (주말 제외)
pred_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), 
    periods=5, 
    freq='B'  # Business day (월~금)
)

pred_prices = predict_future_days(model, last_sequence, scaler)

# 8. 최종 시각화
plt.figure(figsize=(14,7))
plt.plot(data.index, data['Close'], label='Historical Price', color='blue')

# 예측 구간 표시
plt.plot(pred_dates, pred_prices, 
         marker='o', 
         linestyle='--', 
         color='red', 
         label='5-Day Prediction')

plt.title(f'Amazon Stock Price Prediction ({pred_dates[0].date()} ~ {pred_dates[-1].date()})')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

# 예측 결과 표 출력
print("\n[5거래일 예측 결과]")
for date, price in zip(pred_dates, pred_prices):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")