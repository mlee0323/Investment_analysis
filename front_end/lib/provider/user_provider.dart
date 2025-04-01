import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:front_end/userInfo/users.dart';

class UserProvider extends ChangeNotifier {
  late User _userInfo;
  bool _loginStat = false;
  //  bool _loginStat = true; //테스트

  User get userInfo => _userInfo;
  bool get isLogin => _loginStat;

  set userInfo(User userInfo) {
    _userInfo = userInfo;
  }

  set loginStat(bool loginStat) {
    _loginStat = loginStat;
  }

  final Dio _dio = Dio();
  final storage = const FlutterSecureStorage();

  Future<void> login(String username, String password) async {
    _loginStat = false;
    const url = 'http://118.176.174.169:8080/api/users/login';
    final data = {'username': username, 'password': password};

    try {
      final response = await _dio.post(url, data: data);

      if (response.statusCode == 200) {
        print("로그인 성공");

        final authorization = response.headers['authorization']?.first;

        if (authorization == null) {
          print("아이디 또는 비밀번호가 일치하지 않습니다.");
          return;
        }

        final jwt = authorization.replaceFirst('Bearer ', '');
        print("jwt: $jwt");
        await storage.write(key: 'jwt', value: jwt);
        _userInfo = User.fromMap(response.data);
        _loginStat = true;
      } else if (response.statusCode == 403) {
        print('아이디 또는 비밀번호가 일치하지 않습니다.');
      } else {
        print('네트워크 오류 또는 알 수 없는 오류로 로그인에 실패하였습니다.');
      }
    } catch (e) {
      print('로그인 처리 중 에러발생: $e');
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
    const url = 'http://118.176.174.169:8080/api/users/register';
    final data = {
      'username': username,
      'password': password,
      'name': name,
      'email': email,
    };

    try {
      final response = await _dio.post(url, data: data);

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
    final url = 'http://118.176.174.169:8080/api/users/exists/$username';

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
    final url = 'http://118.176.174.169:8080/api/users/$username';

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
}
