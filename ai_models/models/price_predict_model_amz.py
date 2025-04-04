# 1. 필수 라이브러리 설치 (터미널에서 실행)
# !pip install yfinance pandas numpy tensorflow scikit-learn matplotlib

# 2. 데이터 수집 및 전처리
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import tensorflow as tf

# 데이터 다운로드 (3년치)
ticker = "AMZN"
end_date = "2025-03-21"
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=3)  # 3년 전 날짜 자동 계산

data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
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

# 4. 하이브리드 모델 구축 (CNN + LSTM + GRU + Attention)
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Bidirectional, LSTM, GRU, Multiply
from tensorflow.keras.models import Model

def build_hybrid_model(window_size):
    inputs = Input(shape=(window_size, 1))
    
    # CNN-LSTM 경로
    x1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x1 = Dropout(0.2)(x1)
    x1 = Bidirectional(LSTM(100, return_sequences=True))(x1)  # (None, 60, 200)
    
    # GRU 경로
    x2 = Bidirectional(GRU(50, return_sequences=True))(x1)  # (None, 60, 100)
    x2 = BatchNormalization()(x2)
    
    # 어텐션 메커니즘
    attention = Dense(200, activation='tanh')(x2)  # 차원 맞춤
    attention = tf.keras.layers.Reshape((window_size, 200))(attention)  
    
    # 특징 융합
    merged = Multiply()([x1, attention])  # (None, 60, 200) * (None, 60, 200)
    merged = Bidirectional(GRU(30))(merged)
    
    outputs = Dense(1)(merged)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_hybrid_model(window_size)

# 5. 모델 훈련 및 조기 종료 설정
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 6. 미래 5거래일 예측 함수 정의
def predict_future_days(model, last_sequence, scaler, pred_days=5):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(pred_days):
        # 다음 날 예측
        x_input = current_sequence.reshape(1, window_size, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)
        
        # 시퀀스 업데이트 (첫 번째 값 제거 후 새 예측값 추가)
        current_sequence = np.append(current_sequence[1:], pred)
    
    return scaler.inverse_transform([predictions])[0]

# 7. 예측 실행 및 결과 처리
last_date = data.index[-1]
last_sequence = scaled_data[-window_size:]  # 최근 60일 데이터

# 평일만 계산 (주말 제외)
pred_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), 
    periods=5,
    freq='B'  # Business day (월~금만 포함)
)

pred_prices = predict_future_days(model, last_sequence, scaler)

# 실제 목표 가격 (예시 값) - 실제 관측값으로 교체 필요!
actual_prices = [185.50, 187.30, 186.80, 188.90, 190.20]

# 평가 지표 계산
mae = mean_absolute_error(actual_prices[:len(pred_prices)], pred_prices)
rmse = np.sqrt(mean_squared_error(actual_prices[:len(pred_prices)], pred_prices))
mape = mean_absolute_percentage_error(actual_prices[:len(pred_prices)], pred_prices) * 100

# 결과 출력 및 시각화
plt.figure(figsize=(16, 8))

# 히스토리 데이터 플롯
plt.plot(data.index[-200:], data['Close'][-200:], label='Historical Price', color='blue', alpha=0.7)

# 예측 구간 강조 표시
plt.plot(pred_dates[:len(pred_prices)], pred_prices[:len(pred_prices)], 'ro--', linewidth=2,
         markersize=8,
         markeredgecolor='black', label='Predicted Prices')
plt.plot(pred_dates[:len(actual_prices)], actual_prices[:len(actual_prices)], 'g^--', linewidth=2,
         markersize=8,
         markeredgecolor='black', label='Actual Prices')

plt.title(f'Amazon Stock Prediction: {pred_dates[0].strftime("%Y-%m-%d")} to {pred_dates[-1].strftime("%Y-%m-%d")}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(pred_dates[:len(pred_prices)], rotation=45)
plt.grid(linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

print("\n[Performance Metrics]")
print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("\n[Daily Breakdown]")
for date, pred_price in zip(pred_dates[:len(pred_prices)], pred_prices):
    print(f"{date.strftime('%Y-%m-%d')}: Predicted ${pred_price:.2f}")

plt.show()
