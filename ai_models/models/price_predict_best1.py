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
import pickle
import json
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import logging

# TensorFlow 로깅 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no info, 2=no warnings, 3=no errors
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# GPU 사용 가능 여부 확인 및 최적화
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU 사용 가능: {gpus[0]}")
    try:
        # GPU 메모리 설정 (메모리 성장 대신 고정된 메모리 할당 사용)
        for gpu in gpus:
            # 메모리 제한 설정 (90% 사용)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*9)]  # 9GB
            )
    except RuntimeError as e:
        print(f"GPU 메모리 설정 중 오류: {e}")
else:
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

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

# 모든 랜덤 시드 설정
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# 배치 크기 증가 (GPU 메모리에 맞게 조정)
BATCH_SIZE = 128  # 32에서 128로 증가

# 1. 데이터 로드 및 전처리
print("Loading stock data...")
try:
    # 데이터 로딩 시 정렬 보장
    stock_data = pd.read_csv('/kaggle/input/dataset/kospi200_stock_prices_pykrx_20231107_20250321_20250407.csv')
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
    sentiment_data = pd.read_excel('/kaggle/input/dataset/lg_news_finbert_sentiment.xlsx')
    sentiment_data = sentiment_data.sort_values('PubDate').reset_index(drop=True)

    print("Sentiment data columns:", sentiment_data.columns.tolist())
    print("Sentiment data shape:", sentiment_data.shape)
    print("Sentiment data head:\n", sentiment_data.head())
except Exception as e:
    print(f"감성 데이터 로드 실패: {e}")
    raise

print("\nLoading economic indicators data...")
try:
    # 미국 10년물 국채 금리
    treasury = yf.download('^TNX', start='2023-11-07', end='2025-03-21')
    treasury = treasury[['Close']].rename(columns={'Close': 'treasury_10y'})
    
    # 달러 인덱스
    dollar_index = yf.download('DX-Y.NYB', start='2023-11-07', end='2025-03-21')
    dollar_index = dollar_index[['Close']].rename(columns={'Close': 'dollar_index'})
    
    # 원달러 환율
    usdkrw = yf.download('USDKRW=X', start='2023-11-07', end='2025-03-21')
    usdkrw = usdkrw[['Close']].rename(columns={'Close': 'usd_krw'})
    
    # 한국 10년물 국채 금리 (yfinance 사용)
    korean_bond = yf.download('KR10YT=RR', start='2023-11-07', end='2025-03-21')
    korean_bond = korean_bond[['Close']].rename(columns={'Close': 'korean_bond_10y'})
    
    # 모든 경제지표 병합
    economic_data = pd.concat([treasury, dollar_index, usdkrw, korean_bond], axis=1)
    
    # MultiIndex 문제 해결
    if isinstance(economic_data.columns, pd.MultiIndex):
        economic_data.columns = economic_data.columns.get_level_values(0)
    
    # 결측치 처리
    economic_data = economic_data.ffill().bfill()
    
    print("Economic indicators data columns:", economic_data.columns.tolist())
    print("Economic indicators data shape:", economic_data.shape)
    print("Economic indicators data head:\n", economic_data.head())
except Exception as e:
    print(f"경제지표 데이터 로드 실패: {e}")
    raise

# LG전자 데이터만 필터링
lg_data = stock_data[stock_data['종목명'] == 'LG전자'].copy()
print("\nLG data shape:", lg_data.shape)
print("LG data head:\n", lg_data.head())

# 날짜 형식 변환
lg_data['기준일자'] = pd.to_datetime(lg_data['기준일자'])
sentiment_data['PubDate'] = pd.to_datetime(sentiment_data['PubDate'])
economic_data.index = pd.to_datetime(economic_data.index)

# 데이터 병합
merged_data = pd.merge(lg_data, sentiment_data, left_on='기준일자', right_on='PubDate', how='left')
merged_data = pd.merge(merged_data, economic_data, left_on='기준일자', right_index=True, how='left')
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

# 데이터 전처리 개선
def enhanced_preprocessing(df):
    # 가격 변동률 계산
    df['price_change'] = df['현재가'].pct_change()
    df['price_volatility'] = df['price_change'].rolling(window=5).std()
    
    # 거래량 변동률
    df['volume_change'] = df['거래량'].pct_change()
    df['volume_volatility'] = df['volume_change'].rolling(window=5).std()
    
    # 가격 모멘텀
    df['price_momentum'] = df['현재가'] / df['현재가'].rolling(window=5).mean() - 1
    
    # 거래량 모멘텀
    df['volume_momentum'] = df['거래량'] / df['거래량'].rolling(window=5).mean() - 1
    
    # 가격 변동 추세
    df['price_trend'] = df['현재가'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # 이상치 처리 (IQR 방법)
    for col in ['현재가', '거래량', 'price_change', 'volume_change']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 결측치 처리 (최신 pandas 방식)
    df = df.ffill().bfill()
    
    # 감성 데이터 보간
    sentiment_cols = ['finbert_positive', 'finbert_negative', 'finbert_neutral']
    for col in sentiment_cols:
        if col in df.columns:
            # 감성 데이터가 있는 경우에만 보간
            mask = df[col] != 0
            if mask.any():
                df[col] = df[col].interpolate(method='linear')
    
    # 경제 지표 보간
    economic_cols = ['treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y']
    for col in economic_cols:
        if col in df.columns:
            # 경제 지표가 있는 경우에만 보간
            mask = df[col] != 0
            if mask.any():
                df[col] = df[col].interpolate(method='linear')
    
    return df

# 데이터 전처리 적용
merged_data = enhanced_preprocessing(merged_data)

# 스케일링 클래스 개선
class EnhancedPriceScaler:
    def __init__(self):
        self.price_scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # 0-1 범위를 벗어나지 않도록 조정
        self.feature_scaler = MinMaxScaler(feature_range=(0.1, 0.9))

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
        
        # 원래 컬럼 순서 복원
        scaled_df = scaled_df[data_copy.columns]
        
        return scaled_df

    def inverse_transform_price(self, scaled_price):
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)
        
        dummy_data = np.zeros((scaled_price.shape[0], len(self.price_scaler.feature_names_in_)))
        dummy_data[:, 0] = scaled_price.flatten()
        
        unscaled = self.price_scaler.inverse_transform(dummy_data)
        return unscaled[:, 0]

# 손실 함수 개선
def enhanced_weighted_time_mse(y_true, y_pred):
    # 수치적 안정성을 위한 작은 값 추가
    epsilon = 1e-7
    
    # 시간 가중치 조정 (첫날 가중치 강화)
    time_weights = tf.constant([0.6, 0.2, 0.1, 0.07, 0.03], dtype=tf.float32)
    
    # 기본 MSE
    mse_per_step = tf.reduce_mean(tf.square(y_true - y_pred) + epsilon, axis=0)
    
    # 과대 예측 패널티 (첫날 강화)
    overprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_pred - y_true) * tf.constant([25.0, 15.0, 10.0, 8.0, 5.0], dtype=tf.float32)
    )
    
    # 과소 예측 패널티 (첫날 강화)
    underprediction_penalty = tf.reduce_mean(
        tf.maximum(0.0, y_true - y_pred) * tf.constant([15.0, 8.0, 6.0, 4.0, 3.0], dtype=tf.float32)
    )
    
    # 추세 손실 (첫날 강화)
    y_true_diff = y_true[:, 1:] - y_true[:, :-1]
    y_pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
    trend_weights = tf.constant([0.5, 0.3, 0.15, 0.05], dtype=tf.float32)
    trend_loss = tf.reduce_mean(tf.square(y_true_diff - y_pred_diff) * trend_weights + epsilon)
    
    # 방향성 손실 (첫날 강화)
    direction_weights = tf.constant([0.5, 0.3, 0.15, 0.05], dtype=tf.float32)
    direction_loss = tf.reduce_mean(
        tf.square(tf.sign(y_true_diff) - tf.sign(y_pred_diff)) * direction_weights + epsilon
    )
    
    # 가중치 적용
    weighted_loss = (
        tf.reduce_sum(mse_per_step * time_weights) +
        0.7 * overprediction_penalty +
        0.5 * underprediction_penalty +
        0.4 * trend_loss +
        0.3 * direction_loss
    )
    return weighted_loss

# 데이터 증강 함수 개선
def augment_data(X, y, noise_level=0.01):
    """데이터에 노이즈를 추가하여 증강"""
    X_aug = X.copy()
    y_aug = y.copy()
    
    # 가우시안 노이즈 추가
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = X_aug + noise
    
    # 시계열 특성 보존을 위한 노이즈 제한
    X_aug = np.clip(X_aug, X.min(), X.max())
    
    return X_aug, y_aug

# 모델 구조 개선
def build_enhanced_model(input_shape, output_days=5):
    inputs = Input(shape=input_shape)
    
    # 1. 입력 정규화 레이어
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
    x = Dropout(0.2)(x)
    
    # 2. Conv1D 레이어
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Dropout(0.2)(x)
    
    # 3. LSTM 레이어
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.1)
    )(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Dropout(0.2)(x)
    
    # 4. Attention 메커니즘
    attention = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Dropout(0.2)(x)
    
    # 5. 출력 레이어
    outputs = []
    for i in range(output_days):
        day_output = TimeDistributed(Dense(32, activation='relu'))(x)
        day_output = BatchNormalization(momentum=0.9, epsilon=1e-5)(day_output)
        day_output = Dropout(0.2)(day_output)
        day_output = Dense(1, activation='sigmoid', name=f'day_{i+1}_output')(day_output[:, -1, :])  # sigmoid 활성화 함수 사용
        outputs.append(day_output)
    
    final_output = tf.keras.layers.Concatenate()(outputs)
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=final_output)
    
    # 학습률 스케줄링
    initial_learning_rate = 0.0001
    
    optimizer = AdamW(
        learning_rate=initial_learning_rate,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss=enhanced_weighted_time_mse,
        metrics=['mae', 'mse']
    )
    
    return model

# 학습 과정 개선
def train_enhanced_model(model, X_train, y_train, X_test, y_test):
    # 콜백 정의
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            min_delta=0.0001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            '/kaggle/working/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.0001 * (0.95 ** (epoch // 3))
        )
    ]
    
    # 데이터 증강 적용
    X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_level=0.01)
    
    # 학습
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    return history

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
economic_features = [
    'treasury_10y', 'dollar_index', 'usd_krw', 'korean_bond_10y'
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
    'MA5', 'BB_PERCENT', 'MACD',
    'treasury_10y', 'usd_krw'  # 단기 예측에 중요한 경제지표
]

long_term_features = price_features + [
    'MA20', 'MA60', 'BB_PERCENT',
    'MACD', 'MACD_HIST', 'VOLUME_MA20',
    'treasury_10y', 'dollar_index', 'korean_bond_10y'  # 장기 예측에 중요한 경제지표
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
class EnhancedPriceScaler:
    def __init__(self):
        self.price_scaler = MinMaxScaler(feature_range=(0.1, 0.9))  # 0-1 범위를 벗어나지 않도록 조정
        self.feature_scaler = MinMaxScaler(feature_range=(0.1, 0.9))

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
        
        # 원래 컬럼 순서 복원
        scaled_df = scaled_df[data_copy.columns]
        
        return scaled_df

    def inverse_transform_price(self, scaled_price):
        if len(scaled_price.shape) == 1:
            scaled_price = scaled_price.reshape(-1, 1)
        
        dummy_data = np.zeros((scaled_price.shape[0], len(self.price_scaler.feature_names_in_)))
        dummy_data[:, 0] = scaled_price.flatten()
        
        unscaled = self.price_scaler.inverse_transform(dummy_data)
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
    scaler = EnhancedPriceScaler()
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

# 앙상블 모델 클래스 개선
class EnsembleModel:
    def __init__(self, input_shape, n_models=3):
        self.models = []
        self.input_shape = input_shape
        self.n_models = n_models
        self.model_weights = None
        
    def build_models(self):
        for i in range(self.n_models):
            model = build_enhanced_model(self.input_shape)
            self.models.append(model)
    
    def train(self, X_train, y_train, X_val, y_val):
        histories = []
        val_losses = []
        
        # 각 모델 학습
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{self.n_models}")
            
            # 데이터 증강 강화
            X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_level=0.01 * (i + 1))
            
            history = train_enhanced_model(model, X_train_aug, y_train_aug, X_val, y_val)
            histories.append(history)
            
            # 검증 손실 저장
            val_loss = min(history.history['val_loss'])
            val_losses.append(val_loss)
        
        # 모델 가중치 계산 (검증 손실 기반)
        val_losses = np.array(val_losses)
        self.model_weights = 1.0 / (val_losses + 1e-7)
        self.model_weights = self.model_weights / np.sum(self.model_weights)
        
        return histories
    
    def predict(self, X):
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred * self.model_weights[i])
        return np.sum(predictions, axis=0)

# 앙상블 모델 사용
ensemble = EnsembleModel(input_shape=(X_train.shape[1], X_train.shape[2]))
ensemble.build_models()
histories = ensemble.train(X_train, y_train, X_test, y_test)

# 예측 수행
predictions = ensemble.predict(X_test)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(histories[0].history['loss'], label='Training Loss')
plt.plot(histories[0].history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(histories[0].history['mae'], label='Training MAE')
plt.plot(histories[0].history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# 모델 저장
for i, model in enumerate(ensemble.models):
    model.save(f'/kaggle/working/stock_prediction_model_{i+1}.keras')  # .h5 대신 .keras 사용
    
# 스케일러 저장
with open(f'/kaggle/working/stock_prediction_scaler_{i+1}.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# 모델 메타데이터 저장
model_metadata = {
    'input_shape': X_train.shape[1:],
    'output_days': 5,
    'price_scaler_params': {
        'scale_': scaler.price_scaler.scale_.tolist(),
        'min_': scaler.price_scaler.min_.tolist(),  # center_ 대신 min_ 사용
        'data_min_': scaler.price_scaler.data_min_.tolist(),
        'data_max_': scaler.price_scaler.data_max_.tolist(),
        'data_range_': scaler.price_scaler.data_range_.tolist()
    }
}
    
with open(f'/kaggle/working/model_metadata_{i+1}.json', 'w') as f:
    json.dump(model_metadata, f)
        
print(f"모델 {i+1} 저장 완료")
print(f"스케일러 {i+1} 저장 완료")
print(f"메타데이터 {i+1} 저장 완료")

# 예측 결과 분석 및 시각화
try:
    # 테스트 데이터에 대한 예측 수행
    predictions = ensemble.predict(X_test)

    # 마지막 예측 결과 가져오기 (가장 최근 예측)
    last_prediction = predictions[-1]

    # 예측 결과를 원래 스케일로 변환
    last_prediction = scaler.inverse_transform_price(last_prediction.reshape(-1, 1)).flatten()

    # 실제 값과 예측 값 비교
    target_dates = ['2025-03-24', '2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28']
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