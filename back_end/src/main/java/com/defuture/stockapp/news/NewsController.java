package com.defuture.stockapp.news;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class NewsController {
	private final NewsService newsService;
	
	public NewsController(NewsService newsService) {
        this.newsService = newsService;
    }
	
	@GetMapping("/news")
    public ResponseEntity<?> searchBlog(@RequestParam("query") String query) {
        String result = newsService.searchNews(query);
        return ResponseEntity.ok(result);
    }
}