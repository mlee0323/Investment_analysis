import 'package:flutter/material.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';

class MarketScreen extends StatefulWidget {
  const MarketScreen({super.key});

  @override
  State<MarketScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<MarketScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("종목추천")),
      body: Center(child: const Text("종목추천 화면")),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 2),
    );
  }
}
