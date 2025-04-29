class Auth {
  String? username;
  String? name;
  String? email;
  String? password;

  Auth({this.username, this.name, this.email, this.password});

  Map<String, dynamic> toMap() {
    return {
      'username': username,
      'name': name,
      'email': email,
      'password': password,
    };
  }

  factory Auth.fromMap(Map<String, dynamic> map) {
    return Auth(
      username: map['username'],
      name: map['name'],
      email: map['email'],
      password: map['password'],
    );
  }
}
