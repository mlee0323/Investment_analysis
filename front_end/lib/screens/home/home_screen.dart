import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/screens/home/latest/latest_screen.dart';
import 'package:front_end/screens/home/news_test_data.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';
import 'package:front_end/widgets/custom_header.dart';
import 'package:front_end/widgets/custom_horizontal_list.dart';
import 'package:front_end/widgets/custom_vertical_list.dart';
import 'package:provider/provider.dart';
import 'package:timeago/timeago.dart' as timeago;

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _MainPageState();
}

class _MainPageState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    timeago.setLocaleMessages('ko', timeago.KoMessages());
  }

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
      appBar: const CustomHeader(
        showLogo: true,
        showUserIcon: true,
        showBackButton: false,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    '최신 뉴스',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.pushNamed(context, '/latest_list');
                    },
                    child: Text(
                      '전체보기→',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w400,
                        color: Colors.black,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            CustomHorizontalList(
              newsList: List.from(newsList)
                ..sort((a, b) => b.dateTime.compareTo(a.dateTime)),
              onTap: (news) {
                Navigator.pushNamed(context, '/latest', arguments: news);
              },
            ),
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    '추천 뉴스',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.pushNamed(context, '/recommend_list');
                    },
                    child: Text(
                      '전체보기→',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w400,
                        color: Colors.black,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            CustomVerticalList(
              newsList: List.from(newsList)
                ..sort((a, b) => b.recommendScore.compareTo(a.recommendScore)),
              onTap: (news) {
                Navigator.pushNamed(context, '/recommend', arguments: news);
              },
            ),
          ],
        ),
      ),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 0),
    );
  }
}
