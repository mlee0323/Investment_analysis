package com.defuture.stockapp.assets;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
public class AccountEvaluationResponseDTO {
	private String d2EntBalance; // d2_entra D+2추정예수금
	private String totalEstimate; // tot_est_amt 유가잔고평가
	private String totalPurchase; // tot_pur_amt 총매입금액
	private String profitLoss; // lspft 누적투자손익
	private String profitLossRate; // lspft_rt 누적손익률
	private List<StockItemDTO> stocks; // 종목별계좌평가현황
}