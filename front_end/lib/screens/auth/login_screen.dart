import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/widgets/custom_button.dart';
import 'package:provider/provider.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();

  bool _isPasswordVisible = false;

  TextEditingController _usernameController = TextEditingController();
  TextEditingController _passwordController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    UserProvider userProvider = Provider.of<UserProvider>(
      context,
      listen: false,
    );

    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Padding(
        padding: const EdgeInsets.all(40),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 40),
              Center(
                child: Image(image: AssetImage('images/logo.png'), height: 35),
              ),

              const SizedBox(height: 80),
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
              ),

              TextFormField(
                controller: _passwordController,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return '비밀번호를 입력해주세요.';
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
              ),

              const SizedBox(height: 16),
              CustomButton(
                text: '로그인',
                onPressed: () async {
                  if (!_formKey.currentState!.validate()) {
                    return;
                  }

                  final username = _usernameController.text;
                  final password = _passwordController.text;

                  await userProvider.login(username, password);

                  if (userProvider.isLogin) {
                    print('로그인 성공');
                    Navigator.pop(context);
                    Navigator.pushReplacementNamed(context, '/');
                  }
                  print('로그인 실패');
                },
              ),

              const SizedBox(height: 16),
              CustomButton(
                text: "회원가입",
                backgroundColor: Colors.black87,
                onPressed: () {
                  Navigator.pushNamed(context, "/signin");
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
