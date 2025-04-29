// StockPredictionService.java
@Service
public class StockPredictionService {
    
    private final RestTemplate restTemplate;
    private final String apiUrl = "http://localhost:8000";

    public StockPredictionService(RestTemplateBuilder restTemplateBuilder) {
        this.restTemplate = restTemplateBuilder.build();
    }

    // 지원하는 종목 목록 가져오기
    public List<String> getSupportedStocks() {
        String url = apiUrl + "/stocks";
        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
        return (List<String>) response.getBody().get("supported_stocks");
    }

    // 주가 예측
    public PredictionResponse predictStockPrice(String stockSymbol, List<StockData> stockData) {
        String url = apiUrl + "/predict";
        
        // 요청 객체 생성
        Map<String, Object> request = new HashMap<>();
        request.put("stock_symbol", stockSymbol);
        request.put("stock_data", stockData);
        
        // API 호출
        ResponseEntity<PredictionResponse> response = restTemplate.postForEntity(
            url,
            request,
            PredictionResponse.class
        );
        
        return response.getBody();
    }
}

// 컨트롤러에서 사용 예시
@RestController
@RequestMapping("/api/stocks")
public class StockController {
    
    @Autowired
    private StockPredictionService predictionService;
    
    @GetMapping("/supported")
    public ResponseEntity<List<String>> getSupportedStocks() {
        return ResponseEntity.ok(predictionService.getSupportedStocks());
    }
    
    @PostMapping("/{symbol}/predict")
    public ResponseEntity<PredictionResponse> predictStockPrice(
            @PathVariable String symbol,
            @RequestBody List<StockData> stockData) {
        return ResponseEntity.ok(predictionService.predictStockPrice(symbol, stockData));
    }
}