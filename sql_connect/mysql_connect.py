import pymysql

# ==========================================
# ⚙️ [사용자 설정 영역] MySQL 데이터베이스 연결 정보
# 본인의 MySQL 데이터베이스 환경에 맞게 아래의 값을 수정해 주세요.
# ==========================================
DB_CONFIG = {
    'host': 'localhost',         # 데이터베이스 서버 주소 (예: '127.0.0.1')
    'port': 3306,                # 포트 번호 (MySQL 기본값: 3306)
    'user': 'root',              # 데이터베이스 사용자 이름
    'password': 'your_password',  # 사용자 비밀번호
    'database': 'ampi_db',        # 연결할 데이터베이스 이름
    'charset': 'utf8mb4'         # 문자 인코딩 설정
}
# ==========================================

def get_connection():
    """
    MySQL 데이터베이스에 연결하고 연결 객체(Connection)를 반환합니다.
    """
    try:
        connection = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            charset=DB_CONFIG['charset'],
            cursorclass=pymysql.cursors.DictCursor  # 데이터를 딕셔너리 형태로 받음
        )
        return connection
    except pymysql.MySQLError as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return None


def create_history_table():
    """
    딥러닝 모델의 학습 기록(History)을 저장할 테이블을 생성합니다 (예시).
    """
    connection = get_connection()
    if not connection:
        return
        
    try:
        with connection.cursor() as cursor:
            # 테이블 생성 SQL 쿼리
            sql = """
            CREATE TABLE IF NOT EXISTS training_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(50) NOT NULL,
                best_val_accuracy FLOAT,
                epochs INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(sql)
        connection.commit()
        print("✅ training_history 테이블이 준비되었습니다.")
    except pymysql.MySQLError as e:
        print(f"❌ 테이블 생성 실패: {e}")
    finally:
        connection.close()


def insert_training_log(model_name, best_val_accuracy, epochs):
    """
    학습 결과를 데이터베이스의 training_history 테이블에 저장합니다.
    """
    connection = get_connection()
    if not connection:
        return
        
    try:
        with connection.cursor() as cursor:
            # 데이터 삽입 SQL 쿼리
            sql = """
            INSERT INTO training_history (model_name, best_val_accuracy, epochs) 
            VALUES (%s, %s, %s);
            """
            cursor.execute(sql, (model_name, best_val_accuracy, epochs))
        connection.commit()
        print(f"💾 [{model_name}] 학습 결과가 DB에 성공적으로 저장되었습니다.")
    except pymysql.MySQLError as e:
        print(f"❌ 데이터 저장 실패: {e}")
    finally:
        connection.close()


if __name__ == '__main__':
    print("------------------------------------------")
    print("🔍 MySQL 연결 및 테이블 테스트를 시작합니다.")
    print("------------------------------------------")
    
    # 1. 데이터베이스 연결 및 테이블 생성
    create_history_table()
    
    # 2. 샘플 데이터 저장 테스트 (필요 시 주석 해제 후 테스트)
    # insert_training_log(model_name='CNN-TCN', best_val_accuracy=98.52, epochs=15)
    
    print("------------------------------------------")
