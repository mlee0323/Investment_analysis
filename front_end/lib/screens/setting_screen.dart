import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/screens/survey/result_screen.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';
import 'package:front_end/widgets/custom_button.dart';
import 'package:front_end/widgets/custom_header.dart';
import 'package:provider/provider.dart';

class SettingScreen extends StatefulWidget {
  const SettingScreen({super.key});

  @override
  State<SettingScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<SettingScreen> {
  @override
  Widget build(BuildContext context) {
    UserProvider userProvider = Provider.of<UserProvider>(
      context,
      listen: true,
    );
    return Scaffold(
      appBar: const CustomHeader(
        showLogo: true,
        showUserIcon: false,
        showBackButton: false,
      ),
      body: Padding(
        padding: const EdgeInsets.all(40),
        child: Center(
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: 450),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CustomButton(
                  text: '로그아웃',
                  isFullWidth: true,
                  backgroundColor: Color(0xff0FF6B6B),
                  color: Colors.white,
                  onPressed: () async {
                    userProvider.logout();
                    Navigator.pushReplacementNamed(context, '/');
                  },
                ),
                const SizedBox(height: 20),
                CustomButton(
                  text: '설문조사 시작하기',
                  isFullWidth: true,
                  backgroundColor: const Color(0xff1E90FF),
                  color: Colors.white,
                  onPressed: () {
                    Navigator.pushNamed(context, '/survey');
                  },
                ),
                const SizedBox(height: 20),
                CustomButton(
                  text: '설문결과 보기',
                  isFullWidth: true,
                  backgroundColor: const Color(0xff1E90FF),
                  color: Colors.white,
                  onPressed: () {
                    final investInfo = userProvider.investInfo;
                    if (investInfo != null) {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => ResultScreen(result: investInfo),
                        ),
                      );
                    } else {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text("설문 결과가 없습니다. 먼저 설문을 진행해주세요.")),
                      );
                    }
                  },
                ),
              ],
            ),
          ),
        ),
      ),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 3),
    );
  }
}
