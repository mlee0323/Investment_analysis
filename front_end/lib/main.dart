import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/screens/auth/login_screen.dart';
import 'package:front_end/screens/auth/signin_screen.dart';
import 'package:front_end/screens/market_screen.dart';
import 'package:front_end/screens/portfolio_screen.dart';
import 'package:front_end/screens/setting_screen.dart';
import 'screens/home_screen.dart';
import 'package:provider/provider.dart';

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (context) => UserProvider(),
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Investment Analysis',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      debugShowCheckedModeBanner: false,
      initialRoute: "/",
      onGenerateRoute: (settings) {
        switch (settings.name) {
          case '/':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => HomeScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/signin':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => SigninScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/login':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => LoginScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/portfolio':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => PortfolioScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/market':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => MarketScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/setting':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => SettingScreen(),
              transitionDuration: Duration(seconds: 0),
            );
        }
      },
    );
  }
}
