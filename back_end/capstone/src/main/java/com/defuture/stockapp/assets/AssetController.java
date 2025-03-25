package com.defuture.stockapp.assets;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/account")
public class AssetController {

    private final AssetService assetService;

    public AssetController(AssetService assetService) {
    	this.assetService = assetService;
    }

    @GetMapping("/evaluation")
    public ResponseEntity<String> getAccountEvaluation() { //@RequestHeader("Authorization") String token
    	String accessToken = assetService.getAccessToken();
    	System.out.println("🔹 토큰 응답 JSON: " + accessToken);
        String response = assetService.getAccountEvaluation(accessToken); //.replace("Bearer ", "")
        return ResponseEntity.ok(response);
    }
}

