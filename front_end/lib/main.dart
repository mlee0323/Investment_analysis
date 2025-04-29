import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/screens/auth/login_screen.dart';
import 'package:front_end/screens/auth/signup_screen.dart';
import 'package:front_end/screens/home/latest/latest_list_screen.dart';
import 'package:front_end/screens/home/latest/latest_screen.dart';
import 'package:front_end/screens/home/news_test_data.dart';
import 'package:front_end/screens/home/recommend/recommend_list_screen.dart';
import 'package:front_end/screens/home/recommend/recommended_screen.dart';
import 'package:front_end/screens/loadmydata/load_data_screen.dart';
import 'package:front_end/screens/loadmydata/mydata_screen.dart';
import 'package:front_end/screens/market_screen.dart';
import 'package:front_end/screens/portfolio_screen.dart';
import 'package:front_end/screens/setting_screen.dart';
import 'package:front_end/screens/survey/result_screen.dart';
import 'package:front_end/screens/survey/survey_screen.dart';
import 'package:front_end/screens/survey/survey_start_screen.dart';
import 'package:front_end/userInfo/Invest_information.dart';
import 'screens/home/home_screen.dart';
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
        scaffoldBackgroundColor: Color(0xFFF7F7F8),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          focusColor: Colors.white,
          hoverColor: Colors.white,
          hintStyle: TextStyle(color: Color(0xFF91929F)),
          labelStyle: TextStyle(color: Color(0xFF91929F)),
          border: OutlineInputBorder(
            borderSide: BorderSide.none,
            borderRadius: BorderRadius.circular(8),
          ),
          focusedBorder: OutlineInputBorder(
            borderSide: BorderSide(color: Color(0xff3578FF)),
            borderRadius: BorderRadius.circular(8),
          ),
          errorBorder: OutlineInputBorder(
            borderSide: BorderSide(color: Color(0xffFF6B6B)),
            borderRadius: BorderRadius.circular(8),
          ),
          errorStyle: TextStyle(color: Color(0xff0FF6B6B)),
        ),
      ),

      debugShowCheckedModeBanner: false,

      initialRoute: "/survey",
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
                  (context, animation, secondaryAnimation) => SignupScreen(),
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
          case '/survey':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) =>
                      SurveyStartScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/result':
            final args = settings.arguments as InvestInformation;
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) =>
                      ResultScreen(result: args),
              transitionDuration: Duration(seconds: 0),
            );
          case '/mydata':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => MydataScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/loaddata':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) => LoadDataScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/latest_list':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) =>
                      LatestListScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/recommend_list':
            return PageRouteBuilder(
              pageBuilder:
                  (context, animation, secondaryAnimation) =>
                      RecommendListScreen(),
              transitionDuration: Duration(seconds: 0),
            );
          case '/recommend':
            return PageRouteBuilder(
              pageBuilder: (context, animation, secondaryAnimation) {
                final args = settings.arguments;
                return RecommendScreen(news: args as News);
              },
              transitionDuration: Duration(seconds: 0),
            );
          case '/latest':
            return PageRouteBuilder(
              pageBuilder: (context, animation, secondaryAnimation) {
                final args = settings.arguments;
                return LatestScreen(news: args as News);
              },
              transitionDuration: Duration(seconds: 0),
            );
        }
        return null;
      },
    );
  }
}
