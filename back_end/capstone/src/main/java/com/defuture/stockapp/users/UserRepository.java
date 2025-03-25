package com.defuture.stockapp.users;

import java.util.Optional;

import org.springframework.data.mongodb.repository.MongoRepository;

public interface UserRepository extends MongoRepository<UserAccount, String>{
	Optional<UserAccount> findByUserId(String userId);
	boolean existsByUserId(String userId);
	boolean existsByEmail(String email);
}
