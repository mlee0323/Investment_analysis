import 'package:flutter/material.dart';
import 'package:front_end/notifications/snackbar.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/widgets/custom_button.dart';
import 'package:provider/provider.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SigninpageState();
}

class _SigninpageState extends State<SignupScreen> {
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
      backgroundColor: const Color(0xffF7F7F8),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return SingleChildScrollView(
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: Padding(
                padding: const EdgeInsets.all(40),

                child: Form(
                  key: _formKey,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
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
                                  floatingLabelBehavior:
                                      FloatingLabelBehavior.never,
                                  suffixIcon:
                                      _nameController.text.isNotEmpty
                                          ? IconButton(
                                            icon: Icon(Icons.cancel),
                                            color: const Color(0xffA6A6A6),
                                            onPressed: () {
                                              setState(() {
                                                _nameController.clear();
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
                                  floatingLabelBehavior:
                                      FloatingLabelBehavior.never,
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
                                  if (value.length < 6 || value.length > 15) {
                                    return '비밀번호는 6~15글자 이내로 입력해 주세요.';
                                  }
                                  return null;
                                },
                                decoration: InputDecoration(
                                  labelText: '비밀번호',
                                  hintText: '비밀번호',
                                  floatingLabelBehavior:
                                      FloatingLabelBehavior.never,
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
                                  floatingLabelBehavior:
                                      FloatingLabelBehavior.never,
                                  suffixIcon:
                                      _confirmPasswordController.text.isNotEmpty
                                          ? Row(
                                            mainAxisSize: MainAxisSize.min,
                                            children: [
                                              IconButton(
                                                icon: Icon(
                                                  _isConfirmPasswordVisible
                                                      ? Icons.visibility_off
                                                      : Icons.visibility,
                                                ),
                                                color: const Color(0xff6C6C6C),
                                                onPressed: () {
                                                  setState(() {
                                                    _isConfirmPasswordVisible =
                                                        !_isConfirmPasswordVisible;
                                                  });
                                                },
                                              ),
                                              IconButton(
                                                icon: Icon(Icons.cancel),
                                                color: const Color(0xffA6A6A6),
                                                onPressed: () {
                                                  setState(() {
                                                    _confirmPasswordController
                                                        .clear();
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
                                  floatingLabelBehavior:
                                      FloatingLabelBehavior.never,
                                  suffixIcon:
                                      _emailController.text.isNotEmpty
                                          ? IconButton(
                                            icon: Icon(Icons.cancel),
                                            color: const Color(0xffA6A6A6),
                                            onPressed: () {
                                              setState(() {
                                                _emailController.clear();
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
                                    _email = value;
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
                            text: '회원가입',
                            isFullWidth: true,
                            onPressed: () async {
                              if (!_formKey.currentState!.validate()) {
                                return;
                              }

                              final provider = Provider.of<UserProvider>(
                                context,
                                listen: false,
                              );
                              final isDuplicate = await provider
                                  .checkUsernameExists(_username!);

                              if (isDuplicate) {
                                Snackbar(
                                  text: "이미 존재하는 아이디입니다.",
                                  icon: Icons.error,
                                  backgroundColor: Colors.grey,
                                ).showSnackbar(context);
                                return;
                              }

                              try {
                                await context.read<UserProvider>().register(
                                  _username!,
                                  _password!,
                                  _name!,
                                  _email!,
                                );
                                Navigator.pushNamed(context, '/survey');
                              } catch (e) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(content: Text("회원가입 실패: $e")),
                                );
                              }
                            },
                          ),
                        ),
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
            ),
          );
        },
      ),
    );
  }
}
