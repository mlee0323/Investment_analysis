package com.defuture.stockapp.jwt;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;

@Component
public class JwtRequestFilter extends OncePerRequestFilter {

	@Autowired
    private MyUserDetailsService userDetailsService; // 사용자 상세 정보를 조회하는 서비스

    @Autowired
    private JwtUtil jwtUtil; // JWT 관련 유틸리티

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {

    	// 요청 헤더에서 Authorization 값을 가져옴
        final String authorizationHeader = request.getHeader("Authorization");

        String username = null;
        String jwt = null;

        // Authorization 헤더가 "Bearer "로 시작하면 JWT 토큰을 추출
        if (authorizationHeader != null && authorizationHeader.startsWith("Bearer ")) {
            jwt = authorizationHeader.substring(7); // "Bearer " 이후의 토큰 추출
            username = jwtUtil.extractUsername(jwt);
        }

     // 토큰이 유효하고, 아직 SecurityContext에 인증 정보가 없다면 인증 처리
        if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {
            UserDetails userDetails = this.userDetailsService.loadUserByUsername(username);

            if (jwtUtil.validateToken(jwt, userDetails)) {
                // 인증 객체 생성 및 SecurityContext에 설정
                UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(
                        userDetails, null, userDetails.getAuthorities());
                authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                SecurityContextHolder.getContext().setAuthentication(authToken);
            }
        }
        // 필터 체인 계속 진행
        chain.doFilter(request, response);
    }
}
