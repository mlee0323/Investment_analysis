import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';
import 'package:provider/provider.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _MainPageState();
}

class _MainPageState extends State<HomeScreen> {
  @override
  Widget build(BuildContext context) {
    UserProvider userProvider = Provider.of<UserProvider>(
      context,
      listen: true,
    );

    if (!userProvider.isLogin) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (Navigator.canPop(context)) {
          Navigator.pop(context);
        }
        Navigator.pushNamed(context, "/login");
      });
    }

    return Scaffold(
      appBar: AppBar(title: Text("홈")),
      body: Center(child: const Text("홈 화면")),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 0),
    );
  }
}
