import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. 주가 데이터 로드
stock_data = pd.read_csv('kospi200_stock_prices_20231107_20250321_20250329.csv')
stock_data['기준일자'] = pd.to_datetime(stock_data['기준일자'])

# LG 데이터만 필터링
lg_data = stock_data[stock_data['종목명'] == 'LG'].copy()
lg_data = lg_data.sort_values('기준일자')

# 2. 뉴스 감성 데이터 로드
sentiment_data = pd.read_excel('lg_news_finbert_sentiment.xlsx')
sentiment_data['PubDate'] = pd.to_datetime(sentiment_data['PubDate'])

# 3. 일별 감성 점수 집계 및 시차 특성 추가
daily_sentiment = sentiment_data.groupby(pd.to_datetime(sentiment_data['PubDate']).dt.date).agg({
    'finbert_positive': 'mean',
    'finbert_negative': 'mean',
    'finbert_neutral': 'mean'
}).reset_index()
daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['PubDate'])

# 4. 주가 데이터와 감성 데이터 병합
merged_data = pd.merge(lg_data, daily_sentiment, left_on='기준일자', right_on='Date', how='left')

# 5. 시차 특성 및 모멘텀 추가
# 이전 3일의 감성 점수 반영
for lag in [1, 2, 3]:
    merged_data[f'positive_lag_{lag}'] = merged_data['finbert_positive'].shift(lag)
    merged_data[f'negative_lag_{lag}'] = merged_data['finbert_negative'].shift(lag)

# 5일 이동평균으로 뉴스 모멘텀 계산
merged_data['sentiment_momentum'] = (merged_data['finbert_positive'] - merged_data['finbert_negative']).rolling(window=5).mean()

# 결측치 처리 (이동평균으로)
sentiment_cols = ['finbert_positive', 'finbert_negative', 'finbert_neutral', 'sentiment_momentum'] + \
                [f'positive_lag_{i}' for i in [1,2,3]] + \
                [f'negative_lag_{i}' for i in [1,2,3]]
merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(method='ffill').fillna(0.33)

# 6. 특성 선택 및 스케일링
# 주가 데이터와 감성 데이터 분리
price_features = ['시가', '고가', '저가', '현재가', '거래량']
sentiment_features = ['finbert_positive', 'finbert_negative', 'sentiment_momentum']

# 각각의 데이터 정규화
price_scaler = MinMaxScaler()
sentiment_scaler = MinMaxScaler()

price_data = price_scaler.fit_transform(merged_data[price_features])
sentiment_data = sentiment_scaler.fit_transform(merged_data[sentiment_features])

# 데이터 결합
data_scaled = np.hstack((price_data, sentiment_data))

# 7. 시계열 데이터 준비
def create_sequences(data, seq_length, future_days):
    X, y = [], []
    
    if len(data) >= seq_length + future_days:
        for i in range(len(data) - seq_length - future_days + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + future_days, 3])  # 다음 5일의 종가
    
    return np.array(X), np.array(y)

# 시퀀스 길이 설정
seq_length = 110  # 30일로 더 줄임
future_days = 5   # 다음 5일 예측

X, y = create_sequences(data_scaled, seq_length, future_days)
print(f"생성된 시퀀스 수: {len(X)}")

# 8. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 9. LSTM+GRU 하이브리드 모델 구축
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
               recurrent_dropout=0.1))
model.add(Dropout(0.2))
model.add(GRU(128, recurrent_dropout=0.1))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dense(future_days, activation=None))

# 학습률 조정 및 그래디언트 클리핑 적용
optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# 10. 모델 훈련
history = model.fit(X_train, y_train, epochs=200, batch_size=64, 
                   validation_split=0.1, callbacks=[early_stopping], verbose=1)

# 11. 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# 12. 예측
predictions = model.predict(X_test)

# 예측값 역정규화
def inverse_transform_predictions(scaler, predictions, feature_index, n_features):
    predictions_actual = []
    for day in range(predictions.shape[1]):
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, feature_index] = predictions[:, day]
        dummy_transformed = scaler.inverse_transform(dummy)
        predictions_actual.append(dummy_transformed[:, feature_index])
    return np.array(predictions_actual).T

# 예측값 역정규화
predictions_actual = inverse_transform_predictions(price_scaler, predictions, 3, len(price_features))
y_test_actual = inverse_transform_predictions(price_scaler, y_test, 3, len(price_features))

# 13. 다음 5일 예측
def predict_next_days(model, data, seq_length, future_days, price_scaler, sentiment_scaler, price_features, sentiment_features):
    # 최근 seq_length일의 데이터 선택
    recent_price_data = data.tail(seq_length)[price_features].values
    recent_sentiment_data = data.tail(seq_length)[sentiment_features].values
    
    # 데이터 정규화
    recent_price_scaled = price_scaler.transform(recent_price_data)
    recent_sentiment_scaled = sentiment_scaler.transform(recent_sentiment_data)
    
    # 정규화된 데이터 결합
    recent_data_scaled = np.hstack((recent_price_scaled, recent_sentiment_scaled))
    
    # 예측용 입력 데이터 준비
    X_pred = recent_data_scaled.reshape(1, seq_length, len(price_features) + len(sentiment_features))
    
    # 예측
    prediction = model.predict(X_pred)
    
    # 예측값 역정규화
    return inverse_transform_predictions(price_scaler, prediction, 3, len(price_features))[0]

# 다음 5일 예측
print("\n=== 다음 5일 예측 ===")
predictions = predict_next_days(model, merged_data, seq_length, future_days, 
                              price_scaler, sentiment_scaler, price_features, sentiment_features)
for i, pred in enumerate(predictions):
    print(f"{i+1}일 후 예상 주가: {pred:,.0f}원")

# 실제 가격 데이터 추가
actual_prices = [69700, 67500, 67200, 66800, 65700]

# 예측 결과와 실제 가격 비교
print("\n=== 예측 vs 실제 ===")
for i, (pred, actual) in enumerate(zip(predictions, actual_prices)):
    print(f"{i+1}일 후: 예측 {pred:,.0f}원 vs 실제 {actual:,.0f}원")

# 오차 계산
mape = np.mean(np.abs((np.array(actual_prices) - predictions) / np.array(actual_prices))) * 100
print(f"MAPE: {mape:.2f}%")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='실제 주가', marker='o')
plt.plot(predictions, label='예측 주가', marker='x')
plt.title('LG 주가: 예측 vs 실제 (뉴스 감성 분석 통합)')
plt.xlabel('예측 일수')
plt.ylabel('주가')
plt.legend()
plt.grid(True)
plt.show()

