import 'package:front_end/userInfo/auth.dart';

class User {
  String? username;
  String? password;
  String? name;
  String? email;

  User({this.username, this.password, this.name, this.email});

  Map<String, dynamic> toMap() {
    return {
      'username': username,
      'password': password,
      'name': name,
      'email': email,
    };
  }

  factory User.fromMap(Map<String, dynamic> map) {
    return User(
      username: map['username'],
      password: map['password'],
      name: map['name'],
      email: map['email'],
    );
  }
}
