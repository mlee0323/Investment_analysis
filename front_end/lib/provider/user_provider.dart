import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:front_end/userInfo/Invest_information.dart';
import 'package:front_end/userInfo/users.dart';
import 'package:shared_preferences/shared_preferences.dart';

class UserProvider extends ChangeNotifier {
  late User _userInfo;
  late InvestInformation _investInfo;

  bool _loginStat = false;
  //  bool _loginStat = true; //테스트

  User get userInfo => _userInfo;
  InvestInformation get investInfo => _investInfo;
  bool get isLogin => _loginStat;

  String? _loginErrorMessage;
  String? get loginErrorMessage => _loginErrorMessage;

  set userInfo(User userInfo) {
    _userInfo = userInfo;
  }

  setInvestInfo(InvestInformation info) {
    _investInfo = info;
  }

  set loginStat(bool loginStat) {
    _loginStat = loginStat;
  }

  final Dio _dio = Dio();
  final storage = const FlutterSecureStorage();

  Future<void> login(String username, String password) async {
    _loginStat = false;
    _loginErrorMessage = null;

    const url = 'http://10.0.2.2:8080/api/users/login';
    final data = {'username': username, 'password': password};

    try {
      final response = await _dio.post(
        url,
        data: jsonEncode(data),
        options: Options(headers: {'Content-Type': 'application/json'}),
      );

      if (response.statusCode == 200) {
        final jwt = response.data['jwt'];

        if (jwt == null || jwt.isEmpty) {
          _loginErrorMessage = "비밀번호가 틀렸습니다.";
          notifyListeners();
          return;
        }

        print("jwt: $jwt");
        await storage.write(key: 'jwt', value: jwt);

        _userInfo = User.fromMap(response.data);
        _loginStat = true;
      } else if (response.statusCode == 403) {
        _loginErrorMessage = "비밀번호가 틀렸습니다.";
      } else {
        _loginErrorMessage = "로그인 중 오류가 발생했습니다.";
      }
    } catch (e) {
      _loginErrorMessage = "네트워크 오류 또는 서버 응답 실패";
      return;
    }
    notifyListeners();
  }

  Future<void> register(
    String username,
    String password,
    String name,
    String email,
  ) async {
    const url = 'http://10.0.2.2:8080/api/users/register';
    final data = {
      'username': username,
      'password': password,
      'name': name,
      'email': email,
    };

    try {
      final response = await _dio.post(
        url,
        data: jsonEncode(data),
        options: Options(headers: {'Content-Type': 'application/json'}),
      );

      if (response.statusCode == 200) {
        print('회원가입 성공: ${response.data['username']}');
      } else {
        print('회원가입 실패');
      }
    } catch (e) {
      print('회원가입 처리 중 에러발생: $e');
    }
  }

  Future<bool> checkUsernameExists(String username) async {
    final url = 'http://10.0.2.2:8080/api/users/exists/$username';

    try {
      final response = await _dio.get(url);

      if (response.statusCode == 200) {
        return response.data;
      } else {
        throw Exception('아이디 중복 확인 실패');
      }
    } catch (e) {
      print('아이디 중복 확인 중 에러발생: $e');
      return false;
    }
  }

  Future<User> getUserByUsername(String username) async {
    final url = 'http://10.0.2.2:8080/api/users/$username';

    try {
      final response = await _dio.get(url);

      if (response.statusCode == 200) {
        return User.fromMap(response.data);
      } else {
        throw Exception('아이디 존재 확인 실패');
      }
    } catch (e) {
      print('아이디 존재 확인 중 에러발생: $e');
      rethrow;
    }
  }

  Future<InvestInformation?> analyzeInvestmentProfile(
    List<Map<String, int>> responses,
  ) async {
    const url = 'http://10.0.2.2:8080/api/users/profile';

    try {
      final jwt = await storage.read(key: 'jwt');
      if (jwt == null || jwt.isEmpty) {
        print("JWT 토큰이 없습니다.");
        return null;
      }

      final response = await _dio.post(
        url,
        data: jsonEncode({'responses': responses}),
        options: Options(
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $jwt',
          },
        ),
      );

      if (response.statusCode == 200) {
        final decoded =
            response.data is String ? jsonDecode(response.data) : response.data;
        final investInfo = InvestInformation.fromJson(decoded);
        print(
          "서버에서 받은 투자성향 결과: ${investInfo.investmentType}, 점수: ${investInfo.totalScore}",
        );
        return investInfo;
      } else {
        print("투자성향 분석 실패 - 상태코드: ${response.statusCode}");
        return null;
      }
    } catch (e) {
      print("투자성향 분석 중 오류 발생: $e");
      return null;
    }
  }

  Future<void> logout() async {
    try {
      await storage.delete(key: 'jwt');
      _userInfo = User();
      _loginStat = false;
      storage.delete(key: 'username');
      final prefs = await SharedPreferences.getInstance();
      print("로그아웃 성공");
    } catch (e) {
      print("로그아웃 실패 : $e");
    }
    notifyListeners();
  }
}
