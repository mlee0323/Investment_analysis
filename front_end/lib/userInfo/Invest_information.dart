class InvestInformation {
  final double totalScore;
  final String investmentType;

  InvestInformation({required this.totalScore, required this.investmentType});

  factory InvestInformation.fromJson(Map<String, dynamic> json) {
    return InvestInformation(
      totalScore: json['totalScore'].toDouble(),
      investmentType: json['investmentType'],
    );
  }
}
