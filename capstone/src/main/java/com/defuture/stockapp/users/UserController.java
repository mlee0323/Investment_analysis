package com.defuture.stockapp.users;

import jakarta.validation.Valid;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@RestController
@RequestMapping("/api/users")
public class UserController {
	private final UserService userService;
	
	@PostMapping("/register")
	 public ResponseEntity<?> registerUser(@Valid @RequestBody UserDTO dto) {
		if(userService.userExists(dto.getUsername())) {
			return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("이미 존재하는 ID입니다.");
		}else if(userService.emailExists(dto.getEmail())) {
			return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("이미 존재하는 email입니다.");
		}
        UserAccount savedUser = userService.create(dto);
        return ResponseEntity.ok(savedUser);  // JSON 형식으로 응답
    }
	
	@GetMapping("/exists/{userId}")
	public ResponseEntity<Boolean> checkUserExists(@PathVariable("userId") String userId){
		boolean exists = userService.userExists(userId);
		return ResponseEntity.ok(exists);
	}
	
	@GetMapping("/{userId}")
    public ResponseEntity<?> getUser(@PathVariable("userId") String userId) {
        UserAccount user = userService.findByUsername(userId);
        if (user == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("존재하지 않는 ID입니다.");
        }
        return ResponseEntity.ok(user);
    }	
}
