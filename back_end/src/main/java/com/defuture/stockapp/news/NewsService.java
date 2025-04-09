package com.defuture.stockapp.news;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.defuture.stockapp.assets.AccountEvaluationResponseDTO;
import com.defuture.stockapp.assets.StockItemDTO;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class NewsService {
	private final ObjectMapper objectMapper;
	private final RestTemplate restTemplate;
	
	@Value("${naver.appkey}")
	private String appKey;

	@Value("${naver.secretkey}")
	private String secretKey;
	
	public NewsService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
        this.objectMapper = new ObjectMapper();
    }
	
	public NewsResponseDTO searchNews(String query) {
		System.out.println(query);
		String encodedQuery = query;
		/*
		try {
            encodedQuery = URLEncoder.encode(query, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException("검색어 인코딩 실패", e);
        }
        */
		System.out.println(encodedQuery);
		String url = "https://openapi.naver.com/v1/search/news.json?query=" + encodedQuery;
		
		HttpHeaders headers = new HttpHeaders();
        headers.set("X-Naver-Client-Id", appKey);
        headers.set("X-Naver-Client-Secret", secretKey);
        
        HttpEntity<String> entity = new HttpEntity<>(headers);
        
        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, entity, String.class);
        
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
		String lastBuildDate = root.get("lastBuildDate").asText();
		Integer total = root.get("total").asInt();
		
		List<NewsItemDTO> items = new ArrayList<>();
		JsonNode newsArray = root.get("items");

		for (JsonNode item : newsArray) {
			items.add(new NewsItemDTO(item.get("title").asText(), item.get("link").asText(),
					item.get("description").asText()));
		}

		return new NewsResponseDTO(lastBuildDate, total, items);
	}
	
}
