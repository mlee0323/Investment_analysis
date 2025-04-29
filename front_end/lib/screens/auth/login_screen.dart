import 'package:flutter/material.dart';
import 'package:front_end/notifications/snackbar.dart';
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

  String? _username;
  String? _password;

  @override
  Widget build(BuildContext context) {
    UserProvider userProvider = Provider.of<UserProvider>(
      context,
      listen: false,
    );

    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: const Color(0xffF7F7F8),
      body: Padding(
        padding: const EdgeInsets.all(40),
        child: Center(
          child: Form(
            key: _formKey,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Center(
                  child: Image(
                    image: AssetImage('images/logo.png'),
                    height: 35,
                  ),
                ),

                const SizedBox(height: 40),
                Center(
                  child: Container(
                    constraints: BoxConstraints(maxWidth: 450),
                    child: Column(
                      children: [
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
                                      color: const Color(0xffA6A6A6),
                                      onPressed: () {
                                        setState(() {
                                          _usernameController.clear();
                                        });
                                      },
                                    )
                                    : null,
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
                            return null;
                          },
                          decoration: InputDecoration(
                            labelText: '비밀번호',
                            hintText: '비밀번호',
                            floatingLabelBehavior: FloatingLabelBehavior.never,
                            suffixIcon:
                                _passwordController.text.isNotEmpty
                                    ? Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        IconButton(
                                          icon: Icon(
                                            _isPasswordVisible
                                                ? Icons.visibility_off
                                                : Icons.visibility,
                                          ),
                                          color: const Color(0xff6C6C6C),
                                          onPressed: () {
                                            setState(() {
                                              _isPasswordVisible =
                                                  !_isPasswordVisible;
                                            });
                                          },
                                        ),
                                        IconButton(
                                          icon: Icon(Icons.cancel),
                                          color: const Color(0xffA6A6A6),
                                          onPressed: () {
                                            setState(() {
                                              _passwordController.clear();
                                            });
                                          },
                                        ),
                                      ],
                                    )
                                    : null,
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
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                Center(
                  child: ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: 450),
                    child: CustomButton(
                      text: '로그인',
                      isFullWidth: true,
                      onPressed: () async {
                        if (!_formKey.currentState!.validate()) {
                          return;
                        }

                        final username = _usernameController.text;
                        final password = _passwordController.text;

                        await userProvider.login(username, password);

                        if (userProvider.isLogin) {
                          Navigator.pop(context);
                          Navigator.pushReplacementNamed(context, '/');
                        } else {
                          Snackbar(
                            text:
                                userProvider.loginErrorMessage ??
                                "로그인에 실패하였습니다.",
                            icon: Icons.error,
                            backgroundColor: Colors.grey,
                          ).showSnackbar(context);
                        }
                      },
                    ),
                  ),
                ),

                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      "아직 회원이 아니신가요?",
                      style: TextStyle(fontSize: 16, color: Color(0xff91929F)),
                    ),
                    TextButton(
                      onPressed: () {
                        Navigator.pushNamed(context, "/signin");
                      },
                      child: const Text(
                        "회원가입",
                        style: TextStyle(
                          color: const Color(0xff4C4C57),
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          decoration: TextDecoration.none,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
