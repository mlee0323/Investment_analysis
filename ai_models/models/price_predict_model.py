import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization, Multiply, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import math
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS의 경우
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 재현성 설정
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 1. 데이터 로드 및 전처리
print("Loading stock data...")
stock_data = pd.read_csv('kospi200_stock_prices_20231107_20250321_20250403.csv')
print("Stock data columns:", stock_data.columns.tolist())
print("Stock data shape:", stock_data.shape)
print("Stock data head:\n", stock_data.head())

print("\nLoading sentiment data...")
sentiment_data = pd.read_excel('lg_news_finbert_sentiment.xlsx')
print("Sentiment data columns:", sentiment_data.columns.tolist())
print("Sentiment data shape:", sentiment_data.shape)
print("Sentiment data head:\n", sentiment_data.head())

# LG전자 데이터만 필터링
lg_data = stock_data[stock_data['종목명'] == 'LG전자'].copy()
print("\nLG data shape:", lg_data.shape)
print("LG data head:\n", lg_data.head())

# 날짜 형식 변환
lg_data['기준일자'] = pd.to_datetime(lg_data['기준일자'])
sentiment_data['PubDate'] = pd.to_datetime(sentiment_data['PubDate'])

# 데이터 병합
merged_data = pd.merge(lg_data, sentiment_data, left_on='기준일자', right_on='PubDate', how='left')
print("\nMerged data shape:", merged_data.shape)
print("Merged data head:\n", merged_data.head())

# 시차 특성 추가
for lag in [1, 2, 3, 5, 7, 10]:
    merged_data[f'positive_lag_{lag}'] = merged_data['finbert_positive'].shift(lag)
    merged_data[f'negative_lag_{lag}'] = merged_data['finbert_negative'].shift(lag)
    merged_data[f'neutral_lag_{lag}'] = merged_data['finbert_neutral'].shift(lag)

# 이동평균 추가
for window in [5, 10, 20]:
    merged_data[f'price_ma_{window}'] = merged_data['현재가'].rolling(window=window).mean()
    merged_data[f'price_std_{window}'] = merged_data['현재가'].rolling(window=window).std()

# RSI 추가
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

merged_data['RSI'] = calculate_rsi(merged_data['현재가'])

# 결측치 처리
merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')

# 데이터 구조 확인
print("\n[데이터 구조 확인]")
print("컬럼 목록:", merged_data.columns.tolist())
print("데이터 크기:", merged_data.shape)
print("\n처음 5행:")
print(merged_data.head())

# 상관관계 계산을 위한 추가 데이터 - 다른 관련 주식 선택 (예: LG 그룹사)
# --------------------------------------------------
# 실제 데이터에 있는 모든 주식 출력
print("\n[사용 가능한 주식 목록]")
available_stocks = sorted(stock_data['종목명'].unique())
print(available_stocks)

# LG 그룹사 목록 확장
related_stocks = [
    'LG', 'LG전자', 'LG생활건강', 'LG화학', 'LG디스플레이',
    'LG에너지솔루션', 'LG CNS', 'LG헬로비전', 'LG이노텍',
    'LG마그나', 'LG엔시스', 'LG아이서비스', 'LG헬로모바일'
]

# 실제 데이터에 존재하는 LG 그룹사만 필터링
related_stocks = [stock for stock in related_stocks if stock in available_stocks]
print("\n[사용 가능한 LG 그룹사]")
print(related_stocks)

# 상관관계 행렬 계산 함수 수정
def calculate_correlation_matrix(df, stocks, window=60):
    """주식 간 상관관계 행렬 계산 - 가능한 데이터만 사용"""
    # 종가 데이터 추출 및 피벗
    price_data = df[df['종목명'].isin(stocks)].pivot(index='기준일자', columns='종목명', values='현재가')
    
    # 누락된 데이터가 있는 경우 처리
    available_columns = price_data.columns
    print(f"\n[상관관계 계산에 사용된 주식]")
    print(list(available_columns))
    
    # 수익률 계산 (충분한 데이터가 있는 경우)
    if len(available_columns) > 1 and len(price_data) > window:
        returns = price_data.pct_change().dropna()
        # 롤링 상관관계 계산
        corr_matrix = returns.tail(window).corr().abs()
        return corr_matrix
    else:
        print("\n[경고] 상관관계 계산을 위한 데이터가 부족합니다.")
        print(f"- 필요한 주식 수: 2개 이상")
        print(f"- 현재 사용 가능한 주식 수: {len(available_columns)}개")
        print(f"- 필요한 데이터 기간: {window}일")
        print(f"- 현재 데이터 기간: {len(price_data)}일")
        return pd.DataFrame(np.ones((1, 1)), index=[stocks[0]], columns=[stocks[0]])

# 상관관계 행렬 계산 - 오류 처리 추가
try:
    if len(related_stocks) > 0:
        stock_group = stock_data[stock_data['종목명'].isin(related_stocks)].copy()
        corr_matrix = calculate_correlation_matrix(stock_group, related_stocks)
        # 임계값 적용 (0.8 이상만 연결)
        adj_matrix = (corr_matrix >= 0.8).astype(int)
        # 자기 자신과의 연결 설정
        for i in range(len(corr_matrix.index)):
            adj_matrix.iloc[i, i] = 1
    else:
        # 관련 주식이 없으면 단일 항목 행렬 생성
        print("No related stocks found in data. Creating single-item correlation matrix.")
        corr_matrix = pd.DataFrame([[1]], index=['LG'], columns=['LG'])
        adj_matrix = pd.DataFrame([[1]], index=['LG'], columns=['LG'])
except Exception as e:
    print(f"Error in correlation calculation: {e}")
    # 오류 발생 시 기본값으로 단일 항목 행렬 사용
    corr_matrix = pd.DataFrame([[1]], index=['LG'], columns=['LG'])
    adj_matrix = pd.DataFrame([[1]], index=['LG'], columns=['LG'])

print("Correlation matrix shape:", corr_matrix.shape)
print("Adjacency matrix shape:", adj_matrix.shape)

# 2. 특성 선택 및 스케일링
price_features = ['시가', '고가', '저가', '현재가', '거래량']
sentiment_features = ['finbert_positive', 'finbert_negative', 'finbert_neutral']
technical_features = [col for col in merged_data.columns if col.startswith('price_ma_') 
                     or col.startswith('price_std_') 
                     or col == 'RSI']
lag_features = [col for col in merged_data.columns if 'lag_' in col]

all_features = price_features + sentiment_features + technical_features + lag_features

# 스케일링 전에 데이터 확인
print("\n[스케일링 전 데이터 확인]")
print("가격 특성 데이터 크기:", merged_data[price_features].shape)
print("감성 특성 데이터 크기:", merged_data[sentiment_features].shape)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(merged_data[all_features])

# 스케일링된 데이터 확인
print("\n[스케일링 후 데이터 확인]")
print("스케일링된 데이터 크기:", data_scaled.shape)

# 3. 시퀀스 생성
def create_sequences(data, window_size=30, future_days=5):
    """
    시계열 데이터에서 시퀀스와 타겟 생성
    """
    sequences = []
    targets = []
    
    # 현재가 컬럼의 인덱스 찾기 (price_features에서 '현재가'의 위치)
    current_price_idx = price_features.index('현재가')
    
    for i in range(len(data) - window_size - future_days + 1):
        # 입력 시퀀스
        sequence = data[i:(i + window_size)]
        sequences.append(sequence)
        
        # 타겟 (다음 5일의 현재가)
        target = data[(i + window_size):(i + window_size + future_days), current_price_idx]
        targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"생성된 시퀀스 크기: {sequences.shape}")
    print(f"생성된 타겟 크기: {targets.shape}")
    
    return sequences, targets

# 시퀀스 생성
window_size = 30
future_days = 5
X, y = create_sequences(data_scaled, window_size, future_days)

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# MCI-GRU 셀 구현
class MCI_GRU_Cell(Layer):
    def __init__(self, units, num_heads=4, **kwargs):
        super(MCI_GRU_Cell, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        
    def build(self, input_shape):
        self.state_size = self.units
        
        # GRU 게이트 가중치 - 입력용과 상태용으로 분리
        # 입력용 가중치
        self.W_z_x = self.add_weight(shape=(input_shape[-1], self.units), name='W_z_x',
                                   initializer='glorot_uniform')
        self.W_r_x = self.add_weight(shape=(input_shape[-1], self.units), name='W_r_x',
                                   initializer='glorot_uniform')
        self.W_h_x = self.add_weight(shape=(input_shape[-1], self.units), name='W_h_x',
                                   initializer='glorot_uniform')
        
        # 상태용 가중치
        self.W_z_h = self.add_weight(shape=(self.units, self.units), name='W_z_h',
                                   initializer='glorot_uniform')
        self.W_r_h = self.add_weight(shape=(self.units, self.units), name='W_r_h',
                                   initializer='glorot_uniform')
        self.W_h_h = self.add_weight(shape=(self.units, self.units), name='W_h_h',
                                   initializer='glorot_uniform')
        
        # GRU 게이트 편향
        self.b_z = self.add_weight(shape=(self.units,), name='b_z',
                                 initializer='zeros')
        self.b_r = self.add_weight(shape=(self.units,), name='b_r',
                                 initializer='zeros')
        self.b_h = self.add_weight(shape=(self.units,), name='b_h',
                                 initializer='zeros')
        
        # Multi-head Attention 가중치
        self.W_q = self.add_weight(shape=(input_shape[-1], self.units), name='W_q',
                                 initializer='glorot_uniform')
        self.W_k = self.add_weight(shape=(input_shape[-1], self.units), name='W_k',
                                 initializer='glorot_uniform')
        self.W_v = self.add_weight(shape=(input_shape[-1], self.units), name='W_v',
                                 initializer='glorot_uniform')
        
        # 출력 가중치
        self.W_o = self.add_weight(shape=(self.units, self.units), name='W_o',
                                 initializer='glorot_uniform')
        self.b_o = self.add_weight(shape=(self.units,), name='b_o',
                                 initializer='zeros')
        
        super(MCI_GRU_Cell, self).build(input_shape)
        
    def call(self, inputs, states):
        h_prev = states[0]
        
        # GRU 게이트 계산 - 입력과 상태에 대해 별도의 가중치 사용
        z = tf.sigmoid(tf.matmul(inputs, self.W_z_x) + tf.matmul(h_prev, self.W_z_h) + self.b_z)
        r = tf.sigmoid(tf.matmul(inputs, self.W_r_x) + tf.matmul(h_prev, self.W_r_h) + self.b_r)
        
        # Multi-head Attention 계산
        q = tf.matmul(inputs, self.W_q)
        k = tf.matmul(inputs, self.W_k)
        v = tf.matmul(inputs, self.W_v)
        
        # Attention 스코어 계산
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Attention 출력
        attention_output = tf.matmul(attention_weights, v)
        
        # GRU 상태 업데이트 - 입력과 상태에 대해 별도의 가중치 사용
        h_candidate = tf.tanh(tf.matmul(inputs, self.W_h_x) + 
                            tf.matmul(r * h_prev, self.W_h_h) + self.b_h)
        h_new = (1 - z) * h_prev + z * h_candidate
        
        # 최종 출력
        output = tf.matmul(h_new, self.W_o) + self.b_o
        
        return output, [h_new]
    
    
class MCI_GRU(Layer):
    def __init__(self, units, num_heads=4, return_sequences=False, **kwargs):
        super(MCI_GRU, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.return_sequences = return_sequences
        
    def build(self, input_shape):
        self.cell = MCI_GRU_Cell(self.units, self.num_heads)
        self.cell.build(input_shape)
        super(MCI_GRU, self).build(input_shape)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = []
        states = [tf.zeros((batch_size, self.units))]
        
        for t in range(inputs.shape[1]):
            output, states = self.cell(inputs[:, t, :], states)
            outputs.append(output)
        
        outputs = tf.stack(outputs, axis=1)
        
        if not self.return_sequences:
            outputs = outputs[:, -1, :]
            
        return outputs

# 5. MCI-GRU 모델 생성
def build_mci_gru_model(input_shape, hidden_dim=64):
    inputs = Input(shape=input_shape)
    
    # 첫 번째 MCI-GRU 레이어
    x = MCI_GRU(hidden_dim, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 두 번째 MCI-GRU 레이어
    x = MCI_GRU(hidden_dim)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 출력 레이어
    outputs = Dense(5)(x)
    
    return Model(inputs=inputs, outputs=outputs)

# 모델 생성 및 컴파일
model = build_mci_gru_model(input_shape=(window_size, X.shape[2]))
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 6. 모델 훈련
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# 메모리 관리 설정
# GPU 메모리 관리 설정 (사용 가능한 GPU가 있을 경우에만)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU 메모리 증가 설정 성공")
    except Exception as e:
        print(f"GPU 메모리 설정 실패: {e}")
else:
    print("사용 가능한 GPU가 없습니다. CPU로 실행합니다.")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 7. 예측 수행
last_sequence = data_scaled[-window_size:]
last_sequence = np.expand_dims(last_sequence, axis=0)
predictions = model.predict(last_sequence)

# 예측값 역변환
def inverse_transform_predictions(scaler, predictions, feature_index, n_features):
    predictions_actual = []
    for day in range(predictions.shape[1]):
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, feature_index] = predictions[:, day]
        dummy_transformed = scaler.inverse_transform(dummy)
        predictions_actual.append(dummy_transformed[:, feature_index])
    return np.array(predictions_actual).T

predicted_prices = inverse_transform_predictions(
    scaler, predictions, 3, len(all_features)
)[0]

# 8. 결과 시각화
target_dates = ['2025-03-21', '2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27']
target_prices = [69700, 67500, 67200, 66800, 65700]

plt.figure(figsize=(12, 6))
plt.plot(target_dates, target_prices, 'ro-', label='실제 가격')
plt.plot(target_dates, predicted_prices, 'bs--', label='예측 가격')
plt.title('LG 주가 예측 결과 (MCI-GRU 모델)')
plt.xlabel('날짜')
plt.ylabel('주가 (원)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

# 성능 지표 계산
mae = mean_absolute_error(target_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(target_prices, predicted_prices))
mape = np.mean(np.abs((np.array(target_prices) - predicted_prices) / np.array(target_prices))) * 100

print("\n[예측 결과]")
print(f"MAE: {mae:.2f}원")
print(f"RMSE: {rmse:.2f}원")
print(f"MAPE: {mape:.2f}%")

print("\n[날짜별 예측 결과]")
for date, pred, actual in zip(target_dates, predicted_prices, target_prices):
    error = actual - pred
    error_pct = (error / actual) * 100
    print(f"{date}: 예측가 {pred:,.0f}원 | 실제가 {actual:,.0f}원 | 오차 {error:,.0f}원 ({error_pct:.2f}%)")

plt.show()

# 14. 주식간 상관관계 시각화 (추가 분석) - 오류 처리 추가
# --------------------------------------------------
try:
    if corr_matrix.shape[0] > 1:  # 상관관계 행렬이 충분히 크면 시각화
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='상관계수')
        plt.title('LG 관련 주식 간 상관관계')
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)

        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        plt.tight_layout()
except Exception as e:
    print(f"Error in correlation visualization: {e}")
    # 오류 발생 시 상관관계 시각화 생략

plt.show()