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
  "username": "JunOh",
  "password": "testPassword!",
  "name": "준오",
  "email": "jun@example.com"
}
```

#### 📤 응답 예시
```json
{
    "username": "JunOh123",
    "name": "JunOh",
    "email": "jun123@example.com",
    "password": "$2a$10$bVbj3xl039e3dJJ4Si.90eL.4U6GZy372TbrRKkXw78hJmPx3Y27G"
}
```

#### 🔓 인증 필요 여부
> ❌ 인증 불필요 (`permitAll()`)

---

### 2. 🔍 아이디 중복 확인
- **Method:** `GET`
- **URL:** `/api/users/exists/{username}`
- **설명:** 사용자 ID가 이미 존재하는지 확인

#### 📤 응답 예시
```boolean
true
```

#### 🔓 인증 필요 여부
> ❌ 인증 불필요 (`permitAll()`)

---

### 3. 🙋 아이디 존재 확인
- **Method:** `GET`
- **URL:** `/api/users/{username}`
- **설명:** 사용자 ID가 이미 존재하는지 확인

#### 📤 응답 예시
```json
{
    "username": "JunOh123",
    "name": "JunOh",
    "email": "jun123@example.com",
    "password": "$2a$10$bVbj3xl039e3dJJ4Si.90eL.4U6GZy372TbrRKkXw78hJmPx3Y27G"
}
```

#### 🔒 인증 필요 여부
> ❌ 인증 불필요 (`permitAll()`)

---

### 4. 🔐 로그인
- **Method:** `POST`
- **URL:** `/api/users/login`
- **설명:** 사용자 로그인, JWT 토큰 발급

#### 📦 요청 Body
```json
{
  "username": "jun123",
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

## 💼 자산 관련 API

### 5. 💰 자산 평가 정보 조회
- **Method:** `GET`
- **URL:** `/api/assets`
- **설명:** 키움 API를 이용한 자산 평가 정보 조회

#### 📥 요청 헤더
```
Authorization: Bearer {JWT_TOKEN}
```

#### 📤 응답 예시
```json
{
  "d2EntBalance": "000051842599",
  "totalEstimate": "000417180200",
  "totalPurchase": "000416117111",
  "profitLoss": "000000000000",
  "profitLossRate": "0.00",
  "stocks": [
    {
      "name": "메리츠금융지주",
      "quantity": "000000000900",
      "avgPrice": "000000118234",
      "currentPrice": "000000119600",
      "evalAmount": "000106729370",
      "plAmount": "000000318870",
      "plRate": "0.2997"
    },
    ...
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

🛠 작성일: 2025.03.26  
👨‍💻 담당자: 백엔드 개발 - 박준오
