package com.defuture.stockapp.assets;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class AssetService {
	private final ObjectMapper objectMapper;
	
	@Value("${kiwoom.appkey}")
	private String appKey;
	
	@Value("${kiwoom.secretkey}")
    private String secretKey;

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
        this.objectMapper = new ObjectMapper();
    }
    
    public String getAccessToken() {
        String url = baseUrl + tokenEndpoint;  // API URL 설정

        // 요청 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        
        // 요청 바디(JSON 데이터)
        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("grant_type", "client_credentials");
        requestBody.put("appkey", appKey); //kiwoom rest api appkey
        requestBody.put("secretkey", secretKey); //kiwoom rest api secretkey
        
        ResponseEntity<Map> response = restTemplate.postForEntity(url, requestBody, Map.class);
        Map<String, Object> body = response.getBody();
        
        this.accessToken=(String) body.get("token");
        return this.accessToken;
    }

    public AccountEvaluationResponseDTO getAccountEvaluation(String token) {
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
        
        String json = response.getBody();
        JsonNode root;
        try {
            root = objectMapper.readTree(json);
            // 정상 파싱 후 처리
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            throw new RuntimeException("JSON 파싱 오류", e);
        }

        // 필요한 필드만 추출
        String d2_entra = root.get("d2_entra").asText();
        String tot_est_amt = root.get("tot_est_amt").asText();
        String tot_pur_amt = root.get("tot_pur_amt").asText();
        String lspft = root.get("lspft").asText();
        String lspft_rt = root.get("lspft_rt").asText();

        List<StockItemDTO> stocks = new ArrayList<>();
        JsonNode stockArray = root.get("stk_acnt_evlt_prst");

        for (JsonNode stock : stockArray) {
            stocks.add(new StockItemDTO(
                stock.get("stk_nm").asText(),
                stock.get("rmnd_qty").asText(),
                stock.get("avg_prc").asText(),
                stock.get("cur_prc").asText(),
                stock.get("evlt_amt").asText(),
                stock.get("pl_amt").asText(),
                stock.get("pl_rt").asText()
            ));
        }

        return new AccountEvaluationResponseDTO(
                d2_entra, tot_est_amt, tot_pur_amt, lspft, lspft_rt, stocks
            );
    }
}