package com.defuture.stockapp.users;

import lombok.Data;

@Data
public class InvestmentProfileResponseDTO {
    private String username;
    private double totalScore;  // 혹은 Integer, Double 등 적절한 타입 사용
    private String investmentType;
}