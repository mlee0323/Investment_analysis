package com.defuture.stockapp;

import com.defuture.stockapp.jwt.JwtRequestFilter;
import com.defuture.stockapp.jwt.MyUserDetailsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

@Configuration
@EnableWebSecurity
public class SecurityConfig {
	
	@Autowired
    private MyUserDetailsService myUserDetailsService; // 사용자 상세 정보 서비스

    @Autowired
    private JwtRequestFilter jwtRequestFilter; // JWT 요청 필터
    
 // AuthenticationManager 설정: 사용자 정보와 암호 인코더를 등록
    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
        return authenticationConfiguration.getAuthenticationManager();
    }
	
	@Bean
    SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
        	.csrf(csrf -> csrf.disable())
        	.authorizeHttpRequests(authorize -> authorize
                    .requestMatchers("/api/users/login", "/api/users/register", "/api/users/exists/{userId}", "/api/users/{userId}").permitAll()
                    .anyRequest().authenticated()
                )
                // 세션 관리를 stateless로 설정
                .sessionManagement(session -> session
                    .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                )
                .addFilterBefore(jwtRequestFilter, UsernamePasswordAuthenticationFilter.class);
        
        return http.build();
    }
	
	@Bean
    PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
