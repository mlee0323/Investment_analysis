import 'package:flutter/material.dart';
import 'package:front_end/screens/home/news_test_data.dart';
import 'package:timeago/timeago.dart' as timeago;

class CustomHorizontalList extends StatelessWidget {
  final List<News> newsList;
  final void Function(News) onTap;

  const CustomHorizontalList({
    super.key,
    required this.newsList,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final sortedList = List<News>.from(newsList)
      ..sort((a, b) => b.dateTime.compareTo(a.dateTime));

    return SizedBox(
      height: 358,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: newsList.length,
        itemBuilder: (context, index) {
          final news = newsList[index];
          return Container(
            width: 350,
            margin: EdgeInsets.only(
              left: 16,
              right: index == newsList.length - 1 ? 16 : 0,
            ),
            child: GestureDetector(
              onTap: () => onTap(news),
              child: Card(
                color: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(20.0),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: AspectRatio(
                          aspectRatio: 16 / 9,
                          child: Image.network(
                            news.imageUrl,
                            fit: BoxFit.cover,
                            width: double.infinity,
                          ),
                        ),
                      ),
                    ),

                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0),
                      child: Container(
                        decoration: const BoxDecoration(
                          border: Border(
                            bottom: BorderSide(
                              color: Color(0xFFE5E7EB),
                              width: 1.0,
                            ),
                          ),
                        ),

                        padding: const EdgeInsets.symmetric(horizontal: 16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              news.title,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),

                            const SizedBox(height: 12),
                            Text(
                              news.content,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),

                            const SizedBox(height: 12),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  timeago.format(news.dateTime, locale: 'ko'),
                                  style: const TextStyle(color: Colors.grey),
                                ),
                                Text(
                                  news.source,
                                  style: const TextStyle(color: Colors.grey),
                                ),
                              ],
                            ),
                            const SizedBox(height: 12),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
