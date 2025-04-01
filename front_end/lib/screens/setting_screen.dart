import 'package:flutter/material.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';

class SettingScreen extends StatefulWidget {
  const SettingScreen({super.key});

  @override
  State<SettingScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<SettingScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("설정")),
      body: Center(child: const Text("설정 화면")),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 3),
    );
  }
}
