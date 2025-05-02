import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager

# Timescale Cloud 연결 정보
DB_PARAMS = {
    'host': 'g51c8i9urb.qkzqkclz66.tsdb.cloud.timescale.com',
    'port': '31216',
    'database': 'tsdb',
    'user': 'tsdbadmin',
    'password': 'ciyoiu2l3l4ybde0',
    'sslmode': 'require'
}

@contextmanager
def get_db_connection():
    """데이터베이스 연결을 관리하는 컨텍스트 매니저"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

@contextmanager
def get_db_cursor(commit=True):
    """데이터베이스 커서를 관리하는 컨텍스트 매니저"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()

def execute_query(query, params=None, fetch=True):
    """쿼리를 실행하고 결과를 반환하는 함수"""
    with get_db_cursor() as cursor:
        cursor.execute(query, params)
        if fetch and cursor.description:
            return cursor.fetchall()
        return None

def execute_values_query(query, data):
    """execute_values를 사용하여 대량의 데이터를 삽입하는 함수"""
    with get_db_cursor() as cursor:
        execute_values(cursor, query, data)

def execute_many_query(query, data):
    """executemany를 사용하여 여러 데이터를 삽입하는 함수"""
    with get_db_cursor() as cursor:
        cursor.executemany(query, data)

def execute_transaction(queries):
    """여러 쿼리를 하나의 트랜잭션으로 실행하는 함수"""
    with get_db_cursor() as cursor:
        for query, params in queries:
            cursor.execute(query, params) 