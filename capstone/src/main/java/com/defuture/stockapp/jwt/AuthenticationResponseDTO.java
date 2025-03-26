package com.defuture.stockapp.jwt;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class AuthenticationResponseDTO {
	private String jwt;
	
	public AuthenticationResponseDTO(String jwt) {
        this.jwt = jwt;
    }

    public String getJwt() {
        return jwt;
    }
}
