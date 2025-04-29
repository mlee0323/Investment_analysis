import 'package:flutter/material.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';
import 'package:front_end/widgets/custom_header.dart';

class PortfolioScreen extends StatefulWidget {
  const PortfolioScreen({super.key});

  @override
  State<PortfolioScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<PortfolioScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomHeader(
        showLogo: true,
        showUserIcon: true,
        showBackButton: false,
      ),
      body: Center(child: const Text("포트폴리오 화면")),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 1),
    );
  }
}
