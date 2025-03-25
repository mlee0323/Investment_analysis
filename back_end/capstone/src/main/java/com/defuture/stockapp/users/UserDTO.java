package com.defuture.stockapp.users;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class UserDTO {
	@Size(min = 3, max = 25)
    @NotBlank(message = "사용자ID는 필수항목입니다.")
    private String userId;

    @NotBlank(message = "비밀번호는 필수항목입니다.")
    @Size(min = 6, message = "비밀번호는 최소 6자 이상이어야 합니다.")
    private String password;
    
    @NotBlank(message = "이름은 필수항목입니다.")
    private String name;

    @NotBlank(message = "이메일은 필수항목입니다.")
    @Email(message = "올바른 이메일 형식이어야 합니다.")
    private String email;
}
