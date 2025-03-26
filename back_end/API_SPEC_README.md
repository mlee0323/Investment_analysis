# 📚 API 명세서 - Spring Boot 백엔드

> JWT 인증 기반, MongoDB 사용  
> 회원가입, 로그인, 사용자 조회, 자산 평가 등 API 제공

---

## ✅ 인증 관련 API

### 1. 📥 회원가입
- **Method:** `POST`
- **URL:** `/api/users/register`
- **설명:** 사용자 회원가입 처리

#### 📦 요청 Body
```json
{
  "userId": "jun123",
  "password": "securePassword!",
  "name": "준이",
  "email": "jun@example.com"
}
```

#### 📤 응답 예시
```json
{
  "message": "회원가입이 완료되었습니다"
}
```

#### 🔓 인증 필요 여부
> ❌ 인증 불필요 (`permitAll()`)

---

### 2. 🔍 아이디 중복 확인
- **Method:** `GET`
- **URL:** `/api/users/exists/{userId}`
- **설명:** 사용자 ID가 이미 존재하는지 확인

#### 📤 응답 예시
```json
{
  "exists": true
}
```

#### 🔓 인증 필요 여부
> ❌ 인증 불필요 (`permitAll()`)

---

### 3. 🔐 로그인
- **Method:** `POST`
- **URL:** `/api/users/login`
- **설명:** 사용자 로그인, JWT 토큰 발급

#### 📦 요청 Body
```json
{
  "userId": "jun123",
  "password": "securePassword!"
}
```

#### 📤 응답 예시
```json
{
  "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 🔓 인증 필요 여부
> ❌ 인증 불필요 (`permitAll()`)

---

## 👤 사용자 API

### 4. 🙋 내 정보 조회
- **Method:** `GET`
- **URL:** `/api/users/{userId}`
- **설명:** 로그인한 사용자의 정보 반환

#### 📥 요청 헤더
```
Authorization: Bearer {JWT_TOKEN}
```

#### 📤 응답 예시
```json
{
  "userId": "jun123",
  "name": "준이",
  "email": "jun@example.com"
}
```

#### 🔒 인증 필요 여부
> ✅ 인증 필요 (JWT)

---

## 💼 자산 관련 API

### 5. 💰 자산 평가 정보 조회
- **Method:** `GET` or `POST`
- **URL:** `/api/assets`
- **설명:** 키움 API를 이용한 자산 평가 정보 조회

#### 📥 요청 헤더
```
Authorization: Bearer {JWT_TOKEN}
```

#### 📤 응답 예시
```json
{
  "totalAsset": 1340000,
  "items": [
    {
      "stockCode": "005930",
      "stockName": "삼성전자",
      "quantity": 10,
      "currentPrice": 67000
    }
  ]
}
```

#### 🔒 인증 필요 여부
> ✅ 인증 필요 (JWT)

---

## 🧾 비고

- 모든 요청은 `application/json` 헤더를 포함해야 합니다.
- 로그인 후 받은 JWT 토큰은 모든 인증이 필요한 API에 `Authorization` 헤더로 함께 보내야 합니다.
  ```
  Authorization: Bearer <token>
  ```

---

🛠 작성일: 2025.03  
👨‍💻 담당자: 백엔드 개발 - 너