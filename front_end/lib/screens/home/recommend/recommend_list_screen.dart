import 'package:flutter/material.dart';
import 'package:front_end/screens/home/news_test_data.dart';
import 'package:front_end/widgets/custom_bottom_navigation_bar.dart';
import 'package:front_end/widgets/custom_header.dart';
import 'package:front_end/widgets/custom_news_list.dart';

class RecommendListScreen extends StatefulWidget {
  const RecommendListScreen({super.key});

  @override
  State<RecommendListScreen> createState() => _RecommendListScreenState();
}

class _RecommendListScreenState extends State<RecommendListScreen> {
  @override
  Widget build(BuildContext context) {
    final sortedNewsList = [...newsList]
      ..sort((a, b) => b.recommendScore.compareTo(a.recommendScore));

    return Scaffold(
      appBar: const CustomHeader(
        showLogo: false,
        showUserIcon: true,
        showBackButton: true,
        title: "추천뉴스",
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.symmetric(vertical: 8.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CustomNewsList(
                newsList: sortedNewsList,
                onTap: (news) {
                  Navigator.pushNamed(context, '/recommend', arguments: news);
                },
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: CustomBottomNavigationBar(currentIndex: 0),
    );
  }
}
