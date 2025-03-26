package com.defuture.stockapp.assets;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class StockItemDTO {
    private String name;        // stk_nm	종목명
    private String quantity;    // rmnd_qty	보유수량
    private String avgPrice;    // avg_prc	평균단가
    private String currentPrice;// cur_prc	현재가
    private String evalAmount;  // evlt_amt	평가금액
    private String plAmount;    // pl_amt	손익금액
    private String plRate;      // pl_rt	손익률
}