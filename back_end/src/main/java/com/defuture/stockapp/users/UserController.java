package com.defuture.stockapp.users;

import jakarta.validation.Valid;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
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
		if (userService.userExists(dto.getUsername())) {
			return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("이미 존재하는 ID입니다.");
		} else if (userService.emailExists(dto.getEmail())) {
			return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("이미 존재하는 email입니다.");
		}
		UserAccount user = userService.create(dto);
		return ResponseEntity.ok(user); // JSON 형식으로 응답
	}

	@GetMapping("/exists/{username}")
	public ResponseEntity<Boolean> checkUserExists(@PathVariable("username") String username) {
		boolean exists = userService.userExists(username);
		return ResponseEntity.ok(exists);
	}

	@GetMapping("/{username}")
	public ResponseEntity<?> getUser(@PathVariable("username") String username) {
		UserAccount user = userService.findByUsername(username);
		if (user == null) {
			return ResponseEntity.status(HttpStatus.NOT_FOUND).body("존재하지 않는 ID입니다.");
		}
		return ResponseEntity.ok(user);
	}

	@PostMapping("/profile")
	public ResponseEntity<InvestmentProfileResponseDTO> createInvestmentProfile(@RequestBody InvestmentRequestDTO dto) {
		UserDetails userDetails = (UserDetails) SecurityContextHolder.getContext().getAuthentication().getPrincipal();

		String username = userDetails.getUsername();
		List<InvestmentResponseDTO> responses = dto.getResponses();

		UserAccount user = userService.createInvestmentProfile(username, responses);
		InvestmentProfileResponseDTO responseDTO = new InvestmentProfileResponseDTO();
	    responseDTO.setUsername(user.getUsername());
	    responseDTO.setTotalScore(user.getInvestmentScore());
	    responseDTO.setInvestmentType(user.getInvestmentType());
		return ResponseEntity.ok(responseDTO);
	}
}
