package com.defuture.stockapp.users;

import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import jakarta.persistence.Id;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Document(collection = "user_accounts")
public class UserAccount {
	@Id
	private String id;
	
	@Indexed(unique = true)
	private String username;

	private String name;

	@Indexed(unique = true)
	private String email;

	private String password;

	private String investmentType;
	private Double investmentScore;

}