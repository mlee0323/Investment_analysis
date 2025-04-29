import 'package:flutter/material.dart';

class News {
  final String imageUrl;
  final String title;
  final DateTime dateTime;
  final String source;
  final String content;
  final double recommendScore;

  News({
    required this.imageUrl,
    required this.title,
    required this.dateTime,
    required this.source,
    required this.content,
    required this.recommendScore,
  });
}

final List<News> newsList = [
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2023/09/16/14/13/business-8256837_1280.jpg',
    title: '삼성전자, 1분기 실적 발표…예상치 상회 삼성전자, 1분기 실적 발표…예상치 상회',
    dateTime: DateTime.now().subtract(const Duration(hours: 2)),
    source: '연합뉴스',
    content:
        '삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.삼성전자가 2025년 1분기 실적을 발표하며 시장 기대를 웃도는 성과를 기록했다. 반도체 부문이 회복세를 보이면서 전체 실적 개선에 기여했다.',
    recommendScore: 9.8,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2016/12/13/22/15/chart-1905225_1280.jpg',
    title: '코스피 2,700선 돌파…투자 심리 회복 조짐',
    dateTime: DateTime.now().subtract(const Duration(days: 1)),
    source: '조선일보',
    content:
        '국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.국내 증시가 글로벌 경기 회복 기대감에 힘입어 상승세를 보였다. 특히 반도체 및 IT 업종이 강세를 주도했다.',
    recommendScore: 7.2,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2021/01/21/11/09/tesla-5937063_1280.jpg',
    title: '테슬라, AI 로봇 발표로 주가 급등',
    dateTime: DateTime.now().subtract(const Duration(days: 3)),
    source: '매일경제',
    content: '테슬라가 새로운 인공지능 기반 로봇을 공개하며 시장의 주목을 받았다. 이에 따라 테슬라 주가는 7% 이상 급등했다.',
    recommendScore: 10.4,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2014/09/15/17/20/euro-447209_1280.jpg',
    title: '테슬라, AI 로봇 발표로 주가 급등',
    dateTime: DateTime.now().subtract(const Duration(days: 3)),
    source: '매일경제',
    content: '테슬라가 새로운 인공지능 기반 로봇을 공개하며 시장의 주목을 받았다. 이에 따라 테슬라 주가는 7% 이상 급등했다.',
    recommendScore: 8.4,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2014/09/15/17/20/euro-447209_1280.jpg',
    title: '테슬라, AI 로봇 발표로 주가 급등',
    dateTime: DateTime.now().subtract(const Duration(days: 3)),
    source: '매일경제',
    content: '테슬라가 새로운 인공지능 기반 로봇을 공개하며 시장의 주목을 받았다. 이에 따라 테슬라 주가는 7% 이상 급등했다.',
    recommendScore: 8.4,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2014/09/15/17/20/euro-447209_1280.jpg',
    title: '테슬라, AI 로봇 발표로 주가 급등',
    dateTime: DateTime.now().subtract(const Duration(days: 3)),
    source: '매일경제',
    content: '테슬라가 새로운 인공지능 기반 로봇을 공개하며 시장의 주목을 받았다. 이에 따라 테슬라 주가는 7% 이상 급등했다.',
    recommendScore: 8.4,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2014/09/15/17/20/euro-447209_1280.jpg',
    title: '테슬라, AI 로봇 발표로 주가 급등',
    dateTime: DateTime.now().subtract(const Duration(days: 3)),
    source: '매일경제',
    content: '테슬라가 새로운 인공지능 기반 로봇을 공개하며 시장의 주목을 받았다. 이에 따라 테슬라 주가는 7% 이상 급등했다.',
    recommendScore: 8.4,
  ),
  News(
    imageUrl:
        'https://cdn.pixabay.com/photo/2014/09/15/17/20/euro-447209_1280.jpg',
    title: '테슬라, AI 로봇 발표로 주가 급등',
    dateTime: DateTime.now().subtract(const Duration(days: 3)),
    source: '매일경제',
    content: '테슬라가 새로운 인공지능 기반 로봇을 공개하며 시장의 주목을 받았다. 이에 따라 테슬라 주가는 7% 이상 급등했다.',
    recommendScore: 8.4,
  ),
];
