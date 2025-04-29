import 'package:flutter/material.dart';
import 'package:front_end/provider/user_provider.dart';
import 'package:front_end/screens/survey/question_test_list.dart';
import 'package:front_end/screens/survey/result_screen.dart';
import 'package:front_end/screens/survey/survey_screen.dart';
import 'package:front_end/widgets/custom_button.dart';
import 'package:front_end/widgets/custom_header.dart';
import 'package:provider/provider.dart';

class SurveyStartScreen extends StatefulWidget {
  const SurveyStartScreen({super.key});

  @override
  State<SurveyStartScreen> createState() => _SurveyStartScreenState();
}

class _SurveyStartScreenState extends State<SurveyStartScreen> {
  int questionIndex = 0;
  bool isStarted = false;

  List<Map<String, int>> selectedAnswers = [];

  void nextPressed() async {
    if (questionIndex < questionList.length - 1) {
      setState(() {
        questionIndex++;
      });
    } else if (selectedAnswers.length == questionList.length) {
      final userProvider = Provider.of<UserProvider>(context, listen: false);
      final result = await userProvider.analyzeInvestmentProfile(
        selectedAnswers,
      );

      if (result != null) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
        );
      } else {
        showDialog(
          context: context,
          builder:
              (_) => AlertDialog(
                title: const Text('분석 실패'),
                content: const Text('투자 성향 분석에 실패했습니다.'),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('확인'),
                  ),
                ],
              ),
        );
      }
    }
  }

  void backPressed() {
    if (questionIndex > 0) {
      setState(() {
        questionIndex--;
      });
    }
  }

  void answerSelected(List<Map<String, int>> answers) {
    setState(() {
      for (var answer in answers) {
        int questionId = answer['questionId']!;
        int selectedOption = answer['selectedOption']!;

        int existingIndex = selectedAnswers.indexWhere(
          (element) => element['questionId'] == questionId,
        );
        if (existingIndex != -1) {
          selectedAnswers[existingIndex]['selectedOption'] = selectedOption;
        } else {
          selectedAnswers.add({
            'questionId': questionId,
            'selectedOption': selectedOption,
          });
        }
      }
    });
  }

  void resetSurvey() {
    setState(() {
      isStarted = false;
      questionIndex = 0;
      selectedAnswers.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: isStarted ? null : const CustomHeader(showBackButton: true),
      resizeToAvoidBottomInset: false,
      backgroundColor: const Color(0xffF7F7F8),
      body:
          isStarted
              ? SurveyScreen(
                nextPressed: nextPressed,
                backPressed: backPressed,
                questionIndex: questionIndex,
                answerSelected: answerSelected,
                onPageSelected: (index) {
                  setState(() {
                    questionIndex = index;
                  });
                },
                onExitSurvey: resetSurvey,
              )
              : Center(
                child: Container(
                  constraints: const BoxConstraints(
                    maxWidth: 450,
                    maxHeight: 300,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFFFFF),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  padding: const EdgeInsets.symmetric(
                    vertical: 0,
                    horizontal: 40,
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.library_add_check,
                        color: Color(0xffD9E8FF),
                        size: 60,
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        "투자 성향 설문조사",
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 24),
                      const Text(
                        "몇 가지 설문에 답하면 투자 성향에 따라 \n맞춤형 서비스를 제공 받을 수 있어요.",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 15,
                          color: Color(0xff91929F),
                        ),
                      ),
                      const SizedBox(height: 24),
                      SizedBox(
                        width: double.infinity,
                        child: CustomButton(
                          text: '시작하기',
                          isFullWidth: true,
                          onPressed: () {
                            setState(() {
                              isStarted = true;
                            });
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
    );
  }
}
