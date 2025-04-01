package com.defuture.stockapp.jwt;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

import io.jsonwebtoken.security.Keys;
import java.security.Key;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@Component
public class JwtUtil {

	private Key SECRET_KEY = Keys.secretKeyFor(SignatureAlgorithm.HS256);

	public String extractUsername(String token) {
		return extractClaim(token, Claims::getSubject);
	}

	public Date extractExpiration(String token) {
		return extractClaim(token, Claims::getExpiration);
	}

	// 토큰에서 원하는 클레임(claim)을 추출하는 범용 메소드
	public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
		final Claims claims = extractAllClaims(token);
		return claimsResolver.apply(claims);
	}

	// JWT 토큰의 모든 클레임을 추출
	private Claims extractAllClaims(String token) {
		return Jwts.parserBuilder().setSigningKey(SECRET_KEY).build().parseClaimsJws(token).getBody();
	}

	// 토큰이 만료되었는지 확인하는 메소드
	private Boolean isTokenExpired(String token) {
		return extractExpiration(token).before(new Date());
	}

	// 사용자 정보를 기반으로 JWT 토큰을 생성
	public String generateToken(UserDetails userDetails) {
		Map<String, Object> claims = new HashMap<>();
		return createToken(claims, userDetails.getUsername());
	}

	// 토큰 생성 로직 (유효기간 24시간 설정)
	private String createToken(Map<String, Object> claims, String subject) {
		long expirationTime = 1000 * 60 * 60 * 24; // 24시간

		return Jwts.builder().setClaims(claims) // 추가적인 클레임 정보
				.setSubject(subject) // 토큰의 주체 (username)
				.setIssuedAt(new Date(System.currentTimeMillis())) // 발행 시간
				.setExpiration(new Date(System.currentTimeMillis() + expirationTime)) // 만료 시간
				.signWith(SECRET_KEY, SignatureAlgorithm.HS256) // 서명 및 비밀키 설정
				.compact();
	}

	// 토큰의 유효성을 검사 (username 일치 및 만료 여부)
	public Boolean validateToken(String token, UserDetails userDetails) {
		final String username = extractUsername(token);
		return (username.equals(userDetails.getUsername()) && !isTokenExpired(token));
	}
}
