import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization, Multiply, MultiHeadAttention, Layer, TimeDistributed, Lambda, Conv1D
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
import math
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands
import os
from google.colab import drive
drive.mount('/content/drive')


print("TensorFlow 버전:", tf.__version__)

# 한글 폰트 설정
try:
    font_list = [f.name for f in fm.fontManager.ttflist]
    for font in ['NanumBarunGothic', 'NanumGothic', 'Malgun Gothic', 'Gulim']:
        if font in font_list:
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 '{font}' 사용")
            break
    else:
        print("한글 폰트를 찾을 수 없어 기본 폰트 사용")

    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"폰트 설정 오류: {e}")

# 재현성 설정 강화
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# 모든 랜덤 시드 설정
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# GPU 설정 (사용 가능한 경우)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 단일 GPU만 사용
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # GPU 메모리 제한 설정
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024*8)]  # 8GB로 제한
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# 배치 크기 고정
BATCH_SIZE = 32

# 1. 데이터 로드 및 전처리
print("Loading stock data...")
try:
    # 데이터 로딩 시 정렬 보장
    stock_data = pd.read_csv('/content/drive/MyDrive/kospi200_stock_prices_pykrx_20231107_20250321_20250407.csv')
    stock_data = stock_data.sort_values(['종목명', '기준일자']).reset_index(drop=True)

    print("Stock data columns:", stock_data.columns.tolist())
    print("Stock data shape:", stock_data.shape)
    print("Stock data head:\n", stock_data.head())
except Exception as e:
    print(f"주식 데이터 로드 실패: {e}")
    raise

print("\nLoading sentiment data...")
try:
    # 감성 데이터도 정렬 보장
    sentiment_data = pd.read_excel('/content/drive/MyDrive/lg_news_finbert_sentiment.xlsx')
    sentiment_data = sentiment_data.sort_values('PubDate').reset_index(drop=True)

    print("Sentiment data columns:", sentiment_data.columns.tolist())
    print("Sentiment data shape:", sentiment_data.shape)
    print("Sentiment data head:\n", sentiment_data.head())
except Exception as e:
    print(f"감성 데이터 로드 실패: {e}")
    raise

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

# 기술적 지표 추가
def add_technical_indicators(df):
    # RSI
    rsi = RSIIndicator(close=df['현재가'], window=14)
    df['RSI'] = rsi.rsi()

    # MACD
    macd = MACD(close=df['현재가'])
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    df['MACD_HIST'] = macd.macd_diff()

    # 볼린저 밴드
    bbands = BollingerBands(close=df['현재가'], window=20)
    df['BB_UPPER'] = bbands.bollinger_hband()
    df['BB_MIDDLE'] = bbands.bollinger_mavg()
    df['BB_LOWER'] = bbands.bollinger_lband()
    df['BB_PERCENT'] = (df['현재가'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])

    # 이동평균
    df['MA5'] = SMAIndicator(close=df['현재가'], window=5).sma_indicator()
    df['MA20'] = SMAIndicator(close=df['현재가'], window=20).sma_indicator()
    df['MA60'] = SMAIndicator(close=df['현재가'], window=60).sma_indicator()

    # 거래량 지표
    df['VOLUME_MA5'] = SMAIndicator(close=df['거래량'], window=5).sma_indicator()
    df['VOLUME_MA20'] = SMAIndicator(close=df['거래량'], window=20).sma_indicator()
    df['VOLUME_RATIO'] = df['거래량'] / df['VOLUME_MA20']

    # 모멘텀 지표
    df['MOM'] = df['현재가'].diff(10)
    df['ROC'] = ROCIndicator(close=df['현재가'], window=10).roc()

    return df

# 기술적 지표 추가
merged_data = add_technical_indicators(merged_data)

# 결측치 처리
merged_data = merged_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

# 스케일링 클래스 개선
class PriceScaler:
    def __init__(self):
        self.price_scaler = RobustScaler()  # 이상치에 강건한 스케일러 사용
        self.feature_scaler = RobustScaler()

    def fit_transform(self, data, price_cols):
        data_copy = data.copy()

        # 문자열 컬럼과 날짜 컬럼 제외
        exclude_cols = ['기준일자', '종목코드', '종목명', 'Title', 'PubDate', 'finbert_sentiment']
        for col in exclude_cols:
            if col in data_copy.columns:
                data_copy = data_copy.drop(columns=[col])

        # 가격 데이터와 다른 특성 분리
        price_data = data_copy[price_cols]
        other_data = data_copy.drop(columns=price_cols)

        # 각각 스케일링
        scaled_price = self.price_scaler.fit_transform(price_data)
        scaled_other = self.feature_scaler.fit_transform(other_data)

        # 스케일링된 데이터 결합
        scaled_data = np.concatenate([scaled_price, scaled_other], axis=1)
        scaled_df = pd.DataFrame(scaled_data, columns=price_cols + other_data.columns.tolist())

        return scaled_df

    def inverse_transform_price(self, scaled_price):
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)

        dummy_data = np.zeros((scaled_price.shape[0], len(self.price_scaler.feature_names_in_)))
        dummy_data[:, 0] = scaled_price.flatten()

        unscaled = self.price_scaler.inverse_transform(dummy_data)
        return unscaled[:, 0]

# 손실 함수 재설계
def weighted_time_mse(y_true, y_pred):
    # 시간 가중치 조정 (첫날 가중치 증가)
    time_weights = tf.constant([0.6, 0.2, 0.1, 0.07, 0.03], dtype=tf.float32)

    # 기본 MSE
    mse_per_step = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)

    # 과대 예측 패널티 (예측값이 실제값보다 클 때 더 큰 패널티)
    overprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_pred - y_true) * 3.0  # 과대 예측에 3배 패널티
    )

    # 추세 손실 (가중치 증가)
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    trend_loss = tf.reduce_mean(tf.square(y_true_diff - y_pred_diff))

    # 방향성 손실 (가중치 증가)
    direction_loss = tf.reduce_mean(tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)))

    # 가중치 적용
    weighted_loss = (
        tf.reduce_sum(mse_per_step * time_weights) +
        0.15 * overprediction_penalty +  # 과대 예측 패널티 증가
        0.05 * trend_loss +  # 추세 손실 가중치 증가
        0.03 * direction_loss  # 방향성 손실 가중치 증가
    )
    return weighted_loss

# 모델 구조 재설계
def build_model(input_shape, output_days=5):
    inputs = Input(shape=input_shape)

    # 시계열 특성 추출을 위한 Conv1D 레이어 추가
    x = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # GRU 레이어
    x = tf.keras.layers.RNN(
        MCI_GRU_Cell(32),
        return_sequences=True
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Attention 메커니즘 추가
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # 시퀀스 처리
    x = TimeDistributed(Dense(16, activation='relu'))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # 각 예측 일자별 독립적인 출력 레이어
    outputs = []
    for i in range(output_days):
        day_output = Dense(16, activation='relu')(x[:, -1, :])
        day_output = BatchNormalization()(day_output)
        day_output = Dense(1, activation='linear', name=f'day_{i+1}_output')(day_output)
        outputs.append(day_output)

    final_output = Concatenate()(outputs)

    # 모델 생성
    model = Model(inputs=inputs, outputs=final_output)

    # 컴파일 (학습률 감소)
    optimizer = AdamW(
        learning_rate=0.000001,  # 학습률 더 감소
        weight_decay=0.03,  # 가중치 감소 더 증가
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )
    model.compile(optimizer=optimizer,
                 loss=weighted_time_mse,
                 metrics=['mae', 'mse'])

    return model

# 데이터 분석
print("\n[데이터 분석]")
# 최근 30일 데이터 요약 통계
recent_data = merged_data.tail(30).describe()
print("최근 30일 데이터 요약:\n", recent_data['현재가'])

# 데이터 크기 확인
print(f"\n최종 데이터 크기: {merged_data.shape}")

# 2. 특성 선택 및 스케일링
# 실제 존재하는 컬럼만 사용
price_features = ['시가', '고가', '저가', '현재가', '거래량']
sentiment_features = [
    'finbert_positive', 'finbert_negative', 'finbert_neutral'
]
technical_features = [
    'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
    'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_PERCENT',
    'MA5', 'MA20', 'MA60',
    'VOLUME_MA5', 'VOLUME_MA20', 'VOLUME_RATIO',
    'MOM', 'ROC'
]

# 단기 예측을 위한 특성과 장기 예측을 위한 특성 구분
short_term_features = price_features + [
    'RSI', 'VOLUME_RATIO', 'MOM', 'ROC',
    'MA5', 'BB_PERCENT', 'MACD'
]

long_term_features = price_features + [
    'MA20', 'MA60', 'BB_PERCENT',
    'MACD', 'MACD_HIST', 'VOLUME_MA20'
]

# 공통 특성
all_features = list(set(short_term_features + long_term_features))

# 스케일링 전에 데이터 확인
print("\n[스케일링 전 데이터 확인]")
print("사용할 모든 특성 개수:", len(all_features))
print("단기 예측 특성 개수:", len(short_term_features))
print("장기 예측 특성 개수:", len(long_term_features))

# 데이터 타입 확인
print("\n[데이터 타입 확인]")
print(merged_data[all_features].dtypes)

# 정상 범위가 아닌 값 확인
print("\n[비정상적인 값 확인]")
for col in all_features:
    non_finite = (~np.isfinite(merged_data[col])).sum()
    if non_finite > 0:
        print(f"- {col}: {non_finite}개의 비정상적인 값 (inf, -inf, NaN)")
        # 비정상적인 값을 0으로 대체
        merged_data[col] = np.nan_to_num(merged_data[col], nan=0.0, posinf=0.0, neginf=0.0)

# 스케일링 및 역스케일링을 위한 클래스 수정
class PriceScaler:
    def __init__(self):
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))  # 스케일링 범위 확장
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform(self, data, price_cols):
        # 데이터 복사
        data_copy = data.copy()

        # 문자열 컬럼과 날짜 컬럼 제외
        exclude_cols = ['기준일자', '종목코드', '종목명', 'Title', 'PubDate', 'finbert_sentiment']
        for col in exclude_cols:
            if col in data_copy.columns:
                data_copy = data_copy.drop(columns=[col])

        # 가격 데이터와 다른 특성 분리
        price_data = data_copy[price_cols]
        other_data = data_copy.drop(columns=price_cols)

        # 각각 스케일링
        scaled_price = self.price_scaler.fit_transform(price_data)
        scaled_other = self.feature_scaler.fit_transform(other_data)

        # 스케일링된 데이터 결합
        scaled_data = np.concatenate([scaled_price, scaled_other], axis=1)

        # 컬럼 이름 복원
        scaled_df = pd.DataFrame(scaled_data, columns=price_cols + other_data.columns.tolist())

        return scaled_df

    def inverse_transform_price(self, scaled_price):
        # 예측값이 1차원인 경우 2차원으로 변환
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)

        # 더미 데이터 생성 (현재가 컬럼만 사용)
        dummy_data = np.zeros((scaled_price.shape[0], len(self.price_scaler.feature_names_in_)))
        dummy_data[:, 0] = scaled_price.flatten()  # 현재가 컬럼에 예측값 할당

        # 역변환 수행
        unscaled = self.price_scaler.inverse_transform(dummy_data)

        # 현재가 컬럼만 반환
        return unscaled[:, 0]

# 데이터 전처리 수정
try:
    # 문자열 컬럼과 날짜 컬럼 저장
    exclude_cols = ['기준일자', '종목코드', '종목명', 'Title', 'PubDate', 'finbert_sentiment']
    excluded_data = merged_data[exclude_cols].copy()

    # 스케일링할 컬럼 선택
    price_cols = ['현재가', '시가', '고가', '저가']
    other_cols = [col for col in merged_data.columns if col not in price_cols + exclude_cols]

    # 스케일러 초기화 및 적용
    scaler = PriceScaler()
    data_scaled = scaler.fit_transform(merged_data, price_cols)

    # 스케일링된 데이터를 DataFrame으로 변환
    scaled_df = pd.DataFrame(data_scaled, columns=price_cols + other_cols)

    # 제외된 컬럼 다시 추가
    for col in exclude_cols:
        scaled_df[col] = excluded_data[col]

    print("\n[스케일링 후 데이터 확인]")
    print("스케일링된 데이터 크기:", scaled_df.shape)
    print("스케일링된 데이터 샘플:\n", scaled_df.head())

    # 시퀀스 생성
    window_size = 110
    future_days = 5

    # 데이터가 충분한지 확인
    if len(scaled_df) > window_size + future_days:
        # 특성 이름 리스트 생성
        feature_names = price_cols + other_cols

        # numpy 배열로 변환
        data_array = scaled_df[feature_names].values

        # 시퀀스 생성
        sequences = []
        targets = []
        dates = []

        # 현재가 컬럼의 인덱스 찾기
        current_price_idx = feature_names.index('현재가')

        for i in range(len(data_array) - window_size - future_days + 1):
            # 입력 시퀀스
            sequence = data_array[i:(i + window_size)]
            sequences.append(sequence)

            # 타겟 (다음 5일의 현재가)
            target = data_array[(i + window_size):(i + window_size + future_days), current_price_idx]
            targets.append(target)

            # 날짜 정보 저장
            dates.append(scaled_df.index[i + window_size])

        X = np.array(sequences)
        y = np.array(targets)

        print(f"생성된 시퀀스 크기: {X.shape}")
        print(f"생성된 타겟 크기: {y.shape}")

        # 4. 데이터 분할
        split_idx = int(len(X) * 0.8)  # 80% 훈련, 20% 테스트

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]

        print(f"훈련 데이터 크기: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"테스트 데이터 크기: X_test={X_test.shape}, y_test={y_test.shape}")
    else:
        print(f"오류: 시퀀스 생성을 위한 데이터가 부족합니다. 필요: {window_size + future_days}일, 현재: {len(scaled_df)}일")
        print("모델 훈련을 건너뜁니다.")
        raise ValueError("데이터 부족")

except Exception as e:
    print(f"데이터 전처리 중 오류 발생: {e}")
    raise

# 3. 개선된 시퀀스 생성 함수
def create_sequences(data, feat_names, window_size=110, future_days=5):
    """
    시계열 데이터에서 시퀀스와 타겟 생성
    """
    sequences = []
    targets = []
    dates = []

    # 현재가 컬럼의 인덱스 찾기
    current_price_idx = feat_names.index('현재가') if '현재가' in feat_names else None

    if current_price_idx is None:
        raise ValueError("'현재가' 특성을 찾을 수 없습니다.")

    # 데이터를 numpy 배열로 변환
    data_array = data[feat_names].values if isinstance(data, pd.DataFrame) else data

    for i in range(len(data_array) - window_size - future_days + 1):
        # 입력 시퀀스
        sequence = data_array[i:(i + window_size)]
        sequences.append(sequence)

        # 타겟 (다음 5일의 현재가)
        target = data_array[(i + window_size):(i + window_size + future_days), current_price_idx]
        targets.append(target)

        # 날짜 정보 저장
        if isinstance(data, pd.DataFrame):
            dates.append(data.index[i + window_size])
        else:
            dates.append(i + window_size)

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"생성된 시퀀스 크기: {sequences.shape}")
    print(f"생성된 타겟 크기: {targets.shape}")

    # NaN 또는 무한대 값 확인
    if np.isnan(sequences).any() or np.isinf(sequences).any():
        print("경고: 시퀀스에 NaN 또는 무한대 값이 있습니다.")
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=1.0, neginf=0.0)

    if np.isnan(targets).any() or np.isinf(targets).any():
        print("경고: 타겟에 NaN 또는 무한대 값이 있습니다.")
        targets = np.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)

    return sequences, targets, dates

# 개선 9: 특성 인덱스 매핑 생성
def create_feature_indices(all_feature_names, short_term_features, long_term_features):
    """특성 이름에 따른 인덱스 매핑 생성"""
    short_term_indices = [all_feature_names.index(f) for f in short_term_features if f in all_feature_names]
    long_term_indices = [all_feature_names.index(f) for f in long_term_features if f in all_feature_names]

    # 공통 인덱스는 단기와 장기 모두에 포함됨
    common_indices = list(set(short_term_indices) & set(long_term_indices))

    return {
        'short_term': short_term_indices,
        'long_term': long_term_indices,
        'common': common_indices
    }

# 특성 인덱스 매핑 생성
feature_indices = create_feature_indices(all_features, short_term_features, long_term_features)

print("\n[특성 인덱스 정보]")
print(f"단기 예측 특성 인덱스 수: {len(feature_indices['short_term'])}")
print(f"장기 예측 특성 인덱스 수: {len(feature_indices['long_term'])}")
print(f"공통 특성 인덱스 수: {len(feature_indices['common'])}")

# MCI-GRU 셀 구현
class MCI_GRU_Cell(Layer):
    def __init__(self, units, num_heads=4, **kwargs):
        super(MCI_GRU_Cell, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.state_size = units  # state_size 속성 추가
        self.output_size = units  # output_size 속성 추가

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # 입력 투영 가중치 (입력 차원을 units 차원으로 변환)
        self.W_input = self.add_weight(shape=(input_dim, self.units),
                                     name='W_input',
                                     initializer='glorot_uniform')

        # GRU 게이트 가중치
        self.W_z = self.add_weight(shape=(self.units, self.units),
                                 name='W_z',
                                 initializer='glorot_uniform')
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                 name='U_z',
                                 initializer='glorot_uniform')
        self.b_z = self.add_weight(shape=(self.units,),
                                 name='b_z',
                                 initializer='zeros')

        self.W_r = self.add_weight(shape=(self.units, self.units),
                                 name='W_r',
                                 initializer='glorot_uniform')
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                 name='U_r',
                                 initializer='glorot_uniform')
        self.b_r = self.add_weight(shape=(self.units,),
                                 name='b_r',
                                 initializer='zeros')

        self.W_h = self.add_weight(shape=(self.units, self.units),
                                 name='W_h',
                                 initializer='glorot_uniform')
        self.U_h = self.add_weight(shape=(self.units, self.units),
                                 name='U_h',
                                 initializer='glorot_uniform')
        self.b_h = self.add_weight(shape=(self.units,),
                                 name='b_h',
                                 initializer='zeros')

        # Attention 가중치
        self.W_q = self.add_weight(shape=(self.units, self.units),
                                 name='W_q',
                                 initializer='glorot_uniform')
        self.W_k = self.add_weight(shape=(self.units, self.units),
                                 name='W_k',
                                 initializer='glorot_uniform')
        self.W_v = self.add_weight(shape=(self.units, self.units),
                                 name='W_v',
                                 initializer='glorot_uniform')

        self.built = True

    def call(self, inputs, states):
        h_prev = states[0]  # 이전 상태

        # 입력 투영
        x = tf.matmul(inputs, self.W_input)

        # Multi-head self-attention
        q = tf.matmul(h_prev, self.W_q)
        k = tf.matmul(h_prev, self.W_k)
        v = tf.matmul(h_prev, self.W_v)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_scores = tf.matmul(q, k, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(attention_scores)
        context = tf.matmul(attention_weights, v)

        # GRU gates with attention
        z = tf.sigmoid(tf.matmul(x, self.W_z) + tf.matmul(context, self.U_z) + self.b_z)
        r = tf.sigmoid(tf.matmul(x, self.W_r) + tf.matmul(context, self.U_r) + self.b_r)
        h_tilde = tf.tanh(tf.matmul(x, self.W_h) + tf.matmul(r * context, self.U_h) + self.b_h)
        h = (1 - z) * context + z * h_tilde

        return h, [h]

    def get_config(self):
        config = super(MCI_GRU_Cell, self).get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config

# 모델 학습
try:
    # 모델 생성
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print(model.summary())

    # 콜백 정의
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-8)
    ]

    # 배치 크기 조정
    BATCH_SIZE = 4  # 배치 크기 더 감소

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,  # 에포크 수 더 증가
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # 학습 결과 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 예측 결과 분석 및 시각화
    try:
        # 테스트 데이터에 대한 예측 수행
        predictions = model.predict(X_test)

        # 마지막 예측 결과 가져오기 (가장 최근 예측)
        last_prediction = predictions[-1]

        # 예측 결과를 원래 스케일로 변환
        last_prediction = scaler.inverse_transform_price(last_prediction.reshape(-1, 1)).flatten()

        # 실제 값과 예측 값 비교
        target_dates = ['2025-03-21', '2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27']
        target_prices = [69700, 67500, 67200, 66800, 65700]

        # 예측 결과 시각화
        plt.figure(figsize=(12, 6))

        # 날짜를 datetime 객체로 변환
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in target_dates]

        # 실제 가격과 예측 가격 플롯
        plt.plot(dates, target_prices, 'b-', label='실제 가격', marker='o')
        plt.plot(dates, last_prediction, 'r--', label='예측 가격', marker='s')

        # 그래프 스타일링
        plt.title('LG전자 주가 예측 결과 (2025년 3월)', fontsize=14)
        plt.xlabel('날짜', fontsize=12)
        plt.ylabel('주가 (원)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # x축 날짜 포맷 설정
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)

        # 오차율 계산 및 표시
        error_rates = [(pred - actual) / actual * 100 for pred, actual in zip(last_prediction, target_prices)]
        for i, (date, error) in enumerate(zip(dates, error_rates)):
            plt.annotate(f'{error:.2f}%',
                        xy=(date, max(last_prediction[i], target_prices[i])),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=10)

        plt.tight_layout()
        plt.show()

        # 예측 결과 상세 분석
        print("\n[예측 결과 분석]")
        print(f"{'날짜':<12} {'실제 가격':>10} {'예측 가격':>10} {'오차율':>8}")
        print("-" * 45)
        for date, actual, pred, error in zip(target_dates, target_prices, last_prediction, error_rates):
            print(f"{date:<12} {actual:>10,d} {pred:>10.0f} {error:>7.2f}%")

        # 전체 예측 성능 지표
        mae = mean_absolute_error(target_prices, last_prediction)
        mse = mean_squared_error(target_prices, last_prediction)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(error_rates))

        print("\n[전체 예측 성능]")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

    except Exception as e:
        print(f"예측 결과 분석 중 오류 발생: {e}")
        raise

    # 모델 학습 후 저장
    try:
        # 모델 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=500,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # 모델 저장
        model.save('/content/drive/MyDrive/stock_prediction_model.h5')
        
        # 모델을 TensorFlow Lite 형식으로 변환
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        # TFLite 모델 저장
        with open('/content/drive/MyDrive/stock_prediction_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # 모델 메타데이터 저장
        import json
        model_metadata = {
            'input_shape': X_train.shape[1:],
            'output_days': 5,
            'price_scaler_params': {
                'scale_': scaler.price_scaler.scale_.tolist(),
                'center_': scaler.price_scaler.center_.tolist()
            }
        }
        
        with open('/content/drive/MyDrive/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f)
        
        print("모델 저장 완료")
        print("TFLite 모델 저장 완료")
        print("메타데이터 저장 완료")

    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
        raise

except Exception as e:
    print(f"모델 학습 중 오류 발생: {e}")
    raise