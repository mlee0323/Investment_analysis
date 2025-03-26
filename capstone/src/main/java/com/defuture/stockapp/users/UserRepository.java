package com.defuture.stockapp.users;

import java.util.Optional;

import org.springframework.data.mongodb.repository.MongoRepository;

public interface UserRepository extends MongoRepository<UserAccount, String>{
	Optional<UserAccount> findByUsername(String username);
	boolean existsByUsername(String username);
	boolean existsByEmail(String email);
}
