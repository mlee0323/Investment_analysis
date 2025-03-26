package com.defuture.stockapp.jwt;

import com.defuture.stockapp.users.UserAccount;
import com.defuture.stockapp.users.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class MyUserDetailsService implements UserDetailsService {

    @Autowired
    private UserRepository userRepository; // MongoDB에서 사용자 정보를 조회하는 Repository

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // username으로 사용자 정보를 조회
    	UserAccount userAccount = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("사용자를 찾을 수 없습니다."));
        // Spring Security에서 사용하기 위한 User 객체 생성 (비밀번호는 암호화되어 있다고 가정)
        return User.withUsername(userAccount.getUsername())
                .password(userAccount.getPassword())
                .authorities("USER")
                .build();
    }
}
