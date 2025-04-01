package com.defuture.stockapp.users;

import java.util.List;
import lombok.Data;

@Data
public class InvestmentRequestDTO {
	private List<InvestmentResponseDTO> responses;
}
