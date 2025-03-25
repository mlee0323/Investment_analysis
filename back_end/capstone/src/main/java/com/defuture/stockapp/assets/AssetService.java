package com.defuture.stockapp.assets;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.util.HashMap;
import java.util.Map;

@Service
public class AssetService {

    private final RestTemplate restTemplate;
    private String accessToken;

    @Value("${securities.api.base-url}")
    private String baseUrl;
    
    @Value("${securities.api.token-endpoint}")
    private String tokenEndpoint;

    @Value("${securities.api.account-endpoint}")
    private String accountEndpoint;

    public AssetService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }
    
    public String getAccessToken() {
        String url = baseUrl + tokenEndpoint;  // API URL 설정

        // 요청 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        // 요청 바디(JSON 데이터)
        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("grant_type", "client_credentials");
        requestBody.put("appkey", "kiwoom rest api appkey"); //kiwoom rest api appkey
        requestBody.put("secretkey", "kiwoom rest api secretkey"); //kiwoom rest api secretkey
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, requestBody, Map.class);
        Map<String, Object> body = response.getBody();
        
        this.accessToken=(String) body.get("token");
        return this.accessToken;
    }

    public String getAccountEvaluation(String token) {
        String url = baseUrl + accountEndpoint;  // API URL 설정

        // 요청 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(token);  // "Authorization: Bearer {token}"
        headers.add("cont-yn", "N");   // 연속조회 여부
        headers.add("next-key", "");   // 연속조회 키
        headers.add("api-id", "kt00004"); // TR명

        // 요청 바디(JSON 데이터)
        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("qry_tp", "0");
        requestBody.put("dmst_stex_tp", "KRX");

        // HTTP 요청 생성
        HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

        // POST 요청 실행
        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, request, String.class);

        return response.getBody(); // 응답 반환
    }
}