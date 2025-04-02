package com.defuture.stockapp.news;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class NewsService {
	
	private final RestTemplate restTemplate;
	
	@Value("${naver.appkey}")
	private String appKey;

	@Value("${naver.secretkey}")
	private String secretKey;
	
	public NewsService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }
	
	public String searchNews(String query) {
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
        return response.getBody();
	}
	
}
