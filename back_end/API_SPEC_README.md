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

### 5. 📥 투자성향 분석
- **Method:** `POST`
- **URL:** `/api/users/profile`
- **설명:** 로그인한 사용자의 투자성향 테스트 결과(응답 정보)를 기반으로, 투자 점수를 계산하고 투자 유형을 결정하여 사용자 계정을 업데이트한 후, 투자성향 반환

#### 📦 요청 Body
> 로그인된 사용자의 정보는 SecurityContext에서 가져오므로, 요청 Body에는 투자 테스트 응답(response) 데이터만 포함합니다.
  
```json
{
  "responses": [
    { "questionId": 1, "selectedOption": 1 },
    { "questionId": 2, "selectedOption": 1 },
    { "questionId": 3, "selectedOption": 1 },
    { "questionId": 4, "selectedOption": 1 },
    { "questionId": 5, "selectedOption": 1 },
    { "questionId": 6, "selectedOption": 1 },
    { "questionId": 7, "selectedOption": 1 }
  ]
}
```

#### 📤 응답 예시
> 프론트엔드에는 민감한 정보(비밀번호, 이메일 등)를 제외한 투자 결과만 전달됩니다.
  
```json
{
  "username": "jun123",
  "totalScore": 52.9,
  "investmentType": "위험중립형"
}
```

#### 🔓 인증 필요 여부
> ✔️ 인증 필요 (JWT Token을 통한 인증 필요)

---

## 💼 자산 관련 API

### 6. 💰 자산 평가 정보 조회
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
  "d2EntBalance": "000051842599",       // D+2 추정예수금 (d2_entra)
  "totalEstimate": "000417180200",      // 유가잔고 평가금액 (tot_est_amt)
  "totalPurchase": "000416117111",      // 총매입금액 (tot_pur_amt)
  "profitLoss": "000000000000",         // 누적 투자손익 (lspft)
  "profitLossRate": "0.00",             // 누적 손익률 (%) (lspft_rt)
  "stocks": [                           // 종목별 계좌평가 현황
    {
      "name": "메리츠금융지주",          // 종목명 (stk_nm)
      "quantity": "000000000900",        // 보유수량 (rmnd_qty)
      "avgPrice": "000000118234",        // 평균단가 (avg_prc)
      "currentPrice": "000000119600",    // 현재가 (cur_prc)
      "evalAmount": "000106729370",      // 평가금액 (evlt_amt)
      "plAmount": "000000318870",        // 손익금액 (pl_amt)
      "plRate": "0.2997"                 // 손익률 (pl_rt)
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
