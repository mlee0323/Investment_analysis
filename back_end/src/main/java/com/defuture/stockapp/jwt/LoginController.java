package com.defuture.stockapp.jwt;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/users")
public class LoginController {

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private MyUserDetailsService userDetailsService;

    @Autowired
    private JwtUtil jwtUtil; // JWT 유틸리티

    // POST /login 엔드포인트 (로그인 요청 처리)
    @PostMapping("/login")
    public ResponseEntity<?> createAuthenticationToken(@RequestBody AuthenticationRequestDTO authenticationRequestDTO) throws Exception {
        try {
            authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                		authenticationRequestDTO.getUsername(), 
                		authenticationRequestDTO.getPassword()
                )
            );
        } catch (BadCredentialsException e) {
            throw new Exception("아이디나 비밀번호가 올바르지 않습니다.", e);
        }

        // 사용자 정보를 조회
        final UserDetails userDetails = userDetailsService.loadUserByUsername(authenticationRequestDTO.getUsername());
        // JWT 토큰 생성
        final String jwt = jwtUtil.generateToken(userDetails);

        // JWT 토큰을 응답으로 전달
        return ResponseEntity.ok(new AuthenticationResponseDTO(jwt));
    }
}
