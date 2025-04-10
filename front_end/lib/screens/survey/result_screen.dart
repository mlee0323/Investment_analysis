import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:front_end/widgets/custom_button.dart';

String getRiskType(double score) {
  if (score < 21) return "안정형";
  if (score < 41) return "안정추구형";
  if (score < 61) return "위험중립형";
  if (score < 81) return "적극투자형";
  return "공격투자형";
}

class ResultScreen extends StatelessWidget {
  final double score;
  const ResultScreen({Key? key, required this.score}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final result = getRiskType(score);

    final barData = [
      MapEntry('안정형', 15.0),
      MapEntry('안정추구형', 20.0),
      MapEntry('위험중립형', 25.0),
      MapEntry('적극투자형', 22.0),
      MapEntry('공격투자형', 18.0),
    ];

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            children: [
              const SizedBox(height: 16),
              const Text(
                "투자 성향 테스트 결과",
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              Text(
                "${score.toStringAsFixed(1)}점 - 당신의 성향은 「$result」입니다.",
                style: const TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              AspectRatio(
                aspectRatio: 1.3,
                child: BarChart(
                  BarChartData(
                    alignment: BarChartAlignment.spaceAround,
                    maxY: 30,
                    barTouchData: BarTouchData(enabled: false),
                    titlesData: FlTitlesData(
                      leftTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 30,
                        ),
                      ),
                      rightTitles: AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      topTitles: AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      bottomTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          getTitlesWidget: (value, meta) {
                            return SideTitleWidget(
                              axisSide: meta.axisSide,
                              space: 8,
                              child: Padding(
                                padding: const EdgeInsets.only(top: 8.0),
                                child: Text(
                                  barData[value.toInt()].key,
                                  style: const TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                    overflow: TextOverflow.visible,
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                              ),
                            );
                          },
                          reservedSize: 50,
                        ),
                      ),
                    ),
                    borderData: FlBorderData(show: false),
                    barGroups:
                        barData.asMap().entries.map((entry) {
                          final index = entry.key;
                          final e = entry.value;
                          final isUserType = e.key == result;

                          return BarChartGroupData(
                            x: index,
                            barRods: [
                              BarChartRodData(
                                toY: e.value,
                                color:
                                    isUserType ? Colors.blue : Colors.grey[300],
                                borderRadius: BorderRadius.circular(4),
                                width: 20,
                              ),
                            ],
                          );
                        }).toList(),
                  ),
                ),
              ),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        decoration: BoxDecoration(
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 8,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        child: Row(
          children: [
            Expanded(
              child: SizedBox(
                height: 47,
                child: OutlinedButton(
                  onPressed: () {
                    Navigator.pushNamedAndRemoveUntil(
                      context,
                      '/survey',
                      (route) => false,
                    );
                  },
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Color(0xFFD1D5DB), width: 1),
                    backgroundColor: Colors.white,
                    foregroundColor: const Color(0xff374151),
                    textStyle: const TextStyle(fontWeight: FontWeight.w500),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: const Text("재설문 하기"),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: SizedBox(
                height: 47,
                child: CustomButton(
                  text: "결과 적용",
                  onPressed: () {
                    Navigator.pushNamed(context, '/mydata');
                  },
                  backgroundColor: const Color(0xffD9E8FF),
                  color: const Color(0xff3578FF),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
