package com.defuture.stockapp.assets;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class AssetController {

	private final AssetService assetService;

	public AssetController(AssetService assetService) {
		this.assetService = assetService;
	}

	@GetMapping("/assets")
	public ResponseEntity<?> getAccountEvaluation() { // @RequestHeader("Authorization") String token
		String accessToken = assetService.getAccessToken();
		AccountEvaluationResponseDTO response = assetService.getAccountEvaluation(accessToken);

		return ResponseEntity.ok(response);
	}
}
