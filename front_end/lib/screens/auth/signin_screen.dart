import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/widgets/custom_button.dart';
import 'package:provider/provider.dart';

class SigninScreen extends StatefulWidget {
  const SigninScreen({super.key});

  @override
  State<SigninScreen> createState() => _SigninpageState();
}

class _SigninpageState extends State<SigninScreen> {
  final _formKey = GlobalKey<FormState>();

  bool _isPasswordVisible = false;
  bool _isConfirmPasswordVisible = false;

  TextEditingController _passwordController = TextEditingController();
  TextEditingController _confirmPasswordController = TextEditingController();
  TextEditingController _usernameController = TextEditingController();
  TextEditingController _nameController = TextEditingController();
  TextEditingController _emailController = TextEditingController();

  String? _username;
  String? _password;
  String? _confirmPassword;
  String? _name;
  String? _email;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(40),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              const SizedBox(height: 40),
              Center(
                child: Image(image: AssetImage('images/logo.png'), height: 35),
              ),

              const SizedBox(height: 80),
              TextFormField(
                controller: _nameController,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '이름을 입력해주세요.';
                  }
                  return null;
                },
                decoration: InputDecoration(
                  labelText: '이름',
                  hintText: '이름',
                  floatingLabelBehavior: FloatingLabelBehavior.never,
                  suffixIcon:
                      _nameController.text.isNotEmpty
                          ? IconButton(
                            icon: Icon(Icons.cancel),
                            color: Colors.black,
                            onPressed: () {
                              setState(() {
                                _nameController.clear();
                              });
                            },
                          )
                          : null,
                  border: OutlineInputBorder(),
                  focusedBorder: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                  helperText: ' ',
                ),
                onChanged: (value) {
                  setState(() {
                    _name = value;
                  });
                },
              ),

              TextFormField(
                controller: _usernameController,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '아이디를 입력해주세요.';
                  }
                  return null;
                },
                decoration: InputDecoration(
                  labelText: '아이디',
                  hintText: '아이디',
                  floatingLabelBehavior: FloatingLabelBehavior.never,
                  suffixIcon:
                      _usernameController.text.isNotEmpty
                          ? IconButton(
                            icon: Icon(Icons.cancel),
                            color: Colors.black,
                            onPressed: () {
                              setState(() {
                                _usernameController.clear();
                              });
                            },
                          )
                          : null,
                  border: OutlineInputBorder(),
                  focusedBorder: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                  helperText: ' ',
                ),
                onChanged: (value) {
                  setState(() {
                    _username = value;
                  });
                },
              ),

              TextFormField(
                controller: _passwordController,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '비밀번호를 입력해주세요.';
                  }
                  if (value.length < 6 || value.length > 15) {
                    return '비밀번호는 6~15글자 이내로 입력해 주세요.';
                  }
                  return null;
                },
                decoration: InputDecoration(
                  labelText: '비밀번호',
                  hintText: '비밀번호',
                  floatingLabelBehavior: FloatingLabelBehavior.never,
                  suffixIcon: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: Icon(
                          _isPasswordVisible
                              ? Icons.visibility_off
                              : Icons.visibility,
                        ),
                        color: Colors.black,
                        onPressed: () {
                          setState(() {
                            _isPasswordVisible = !_isPasswordVisible;
                          });
                        },
                      ),
                      _passwordController.text.isNotEmpty
                          ? IconButton(
                            icon: Icon(Icons.cancel),
                            onPressed: () {
                              setState(() {
                                _passwordController.clear();
                              });
                            },
                          )
                          : SizedBox.shrink(),
                    ],
                  ),
                  border: OutlineInputBorder(),
                  focusedBorder: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                  helperText: ' ',
                ),
                obscureText: !_isPasswordVisible,
                onChanged: (value) {
                  setState(() {
                    _password = value;
                  });
                },
              ),

              TextFormField(
                controller: _confirmPasswordController,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '비밀번호를 확인해주세요.';
                  }
                  if (value != _password) {
                    return '비밀번호가 일치하지 않습니다.';
                  }
                  return null;
                },
                decoration: InputDecoration(
                  labelText: '비밀번호 확인',
                  hintText: '비밀번호 확인',
                  floatingLabelBehavior: FloatingLabelBehavior.never,
                  suffixIcon: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: Icon(
                          _isConfirmPasswordVisible
                              ? Icons.visibility_off
                              : Icons.visibility,
                        ),
                        color: Colors.black,
                        onPressed: () {
                          setState(() {
                            _isConfirmPasswordVisible =
                                !_isConfirmPasswordVisible;
                          });
                        },
                      ),
                      _confirmPasswordController.text.isNotEmpty
                          ? IconButton(
                            icon: Icon(Icons.cancel),
                            onPressed: () {
                              setState(() {
                                _confirmPasswordController.clear();
                              });
                            },
                          )
                          : SizedBox.shrink(),
                    ],
                  ),
                  border: OutlineInputBorder(),
                  focusedBorder: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                  helperText: ' ',
                ),
                obscureText: !_isConfirmPasswordVisible,
                onChanged: (value) {
                  setState(() {
                    _confirmPassword = value;
                  });
                },
              ),

              TextFormField(
                controller: _emailController,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '이메일을 입력해주세요.';
                  }
                  bool emailValid = RegExp(
                    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~]+@[a-zA-Z0-9]+\.[a-zA-Z]+$",
                  ).hasMatch(value);
                  if (!emailValid) {
                    return '이메일 형식에 맞춰 작성해 주세요.';
                  }
                  return null;
                },
                decoration: InputDecoration(
                  labelText: '이메일',
                  hintText: '이메일',
                  floatingLabelBehavior: FloatingLabelBehavior.never,
                  suffixIcon:
                      _emailController.text.isNotEmpty
                          ? IconButton(
                            icon: Icon(Icons.cancel),
                            color: Colors.black,
                            onPressed: () {
                              setState(() {
                                _emailController.clear();
                              });
                            },
                          )
                          : null,
                  border: OutlineInputBorder(),
                  focusedBorder: OutlineInputBorder(
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                  helperText: ' ',
                ),
                onChanged: (value) {
                  setState(() {
                    _email = value;
                  });
                },
              ),

              const SizedBox(height: 16),
              CustomButton(
                text: '회원가입',
                isFullWidth: true,
                onPressed: () async {
                  if (!_formKey.currentState!.validate()) {
                    return;
                  }

                  try {
                    await context.read<UserProvider>().register(
                      _username!,
                      _password!,
                      _name!,
                      _email!,
                    );
                    Navigator.pushNamed(context, '/login');
                  } catch (e) {
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text("회원가입 실패: $e")));
                  }
                },
              ),

              SizedBox(height: 12),
              TextButton(
                onPressed: () {
                  Navigator.pushNamed(context, '/login');
                },
                child: Text(
                  '이미 계정이 있습니다.',
                  style: TextStyle(color: Colors.grey),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
