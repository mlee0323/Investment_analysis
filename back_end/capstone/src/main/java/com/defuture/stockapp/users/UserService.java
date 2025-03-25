package com.defuture.stockapp.users;

import org.springframework.stereotype.Service;
import org.springframework.security.crypto.password.PasswordEncoder;
import java.util.Optional;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Service
public class UserService {
	private final UserRepository userRepository;
	private final PasswordEncoder passwordEncoder;
	
	public UserAccount create(UserDTO form) {
		UserAccount user = new UserAccount();
		user.setUserId(form.getUserId());
		user.setName(form.getName());
		user.setEmail(form.getEmail());
		user.setPassword(passwordEncoder.encode(form.getPassword()));
		
		this.userRepository.save(user);
		return user;
	}
	
	public UserAccount findByUserId(String id) {
		 Optional<UserAccount> user = userRepository.findByUserId(id);
	     return user.orElse(null);
    }
	
	public boolean userExists(String id) {
		return userRepository.existsByUserId(id);
	}
	
	public boolean emailExists(String email) {
		return userRepository.existsByEmail(email);
	}
}
