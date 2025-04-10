import 'package:flutter/material.dart';
import 'package:front_end/screens/survey/question_test_list.dart';
import 'package:front_end/screens/survey/result_screen.dart';
import 'package:front_end/screens/survey/survey_screen.dart';
import 'package:front_end/widgets/custom_button.dart';

class SurveyStartScreen extends StatefulWidget {
  const SurveyStartScreen({super.key});

  @override
  State<SurveyStartScreen> createState() => _SurveyStartScreenState();
}

class _SurveyStartScreenState extends State<SurveyStartScreen> {
  int questionIndex = 0;
  double totalScore = 0.0;
  bool isStarted = false;

  List<List<double>> answerScores = List.generate(
    questionList.length,
    (_) => [],
  );

  void nextPressed() {
    if (questionIndex < questionList.length - 1) {
      setState(() {
        questionIndex++;
      });
    } else if (answerScores.every((element) => element.isNotEmpty)) {
      double finalScore = 0.0;
      for (var scores in answerScores) {
        finalScore += scores.fold(0.0, (sum, s) => sum + s);
      }
      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ResultScreen(score: finalScore)),
      );
    }
  }

  void backPressed() {
    if (questionIndex > 0) {
      setState(() {
        questionIndex--;
      });
    }
  }

  void answerSelected(List<double> scores) {
    setState(() {
      answerScores[questionIndex] = scores;
    });
  }

  void resetSurvey() {
    setState(() {
      isStarted = false;
      questionIndex = 0;
      totalScore = 0.0;
      answerScores = List.generate(questionList.length, (_) => []);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar:
          isStarted
              ? null
              : AppBar(backgroundColor: Colors.white, elevation: 1),
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
