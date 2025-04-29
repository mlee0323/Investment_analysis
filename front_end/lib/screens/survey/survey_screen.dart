import 'package:flutter/material.dart';
import 'package:front_end/screens/survey/question_test_list.dart';
import 'package:front_end/widgets/custom_button.dart';
import 'package:front_end/widgets/custom_header.dart';

class SurveyScreen extends StatefulWidget {
  const SurveyScreen({
    Key? key,
    required this.nextPressed,
    required this.backPressed,
    required this.answerSelected,
    required this.questionIndex,
    required this.onPageSelected,
    required this.onExitSurvey,
  }) : super(key: key);

  final Function nextPressed;
  final Function backPressed;
  final Function(List<Map<String, int>>) answerSelected;
  final int questionIndex;
  final Function(int) onPageSelected;
  final VoidCallback onExitSurvey;

  @override
  State<SurveyScreen> createState() => _SurveyScreenState();
}

class _SurveyScreenState extends State<SurveyScreen> {
  List<int> selectedIndexes = [];

  void handleAnswer(List<Map<String, int>> responses) {
    widget.answerSelected(responses);
  }

  void handleNext() {
    widget.nextPressed();
    setState(() {
      selectedIndexes = [];
    });
  }

  @override
  Widget build(BuildContext context) {
    final question = questionList[widget.questionIndex];
    final int selectOption = (question['selectOption'] as int?) ?? 1;
    ;

    return Scaffold(
      appBar: CustomHeader(
        showBackButton: true,
        showLogo: false,
        showUserIcon: false,
        onPressed: widget.onExitSurvey,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Wrap(
                spacing: 10,
                children: List.generate(questionList.length, (i) {
                  final isCurrent = i == widget.questionIndex;
                  return Container(
                    width: 35,
                    height: 35,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color:
                          isCurrent
                              ? const Color(0xff2D7FFB)
                              : const Color(0xffD9E8FF),
                      shape: BoxShape.circle,
                    ),
                    child: Text(
                      '${i + 1}',
                      style: TextStyle(
                        fontSize: 14,
                        color:
                            isCurrent
                                ? const Color(0xffD9E8FF)
                                : const Color(0xffA6A6A6),
                      ),
                    ),
                  );
                }),
              ),
              const SizedBox(height: 30),
              Center(
                child: Text(
                  question["questionText"],
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 18,
                    color: Colors.black,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 30),
              ...List.generate(question["answers"].length, (index) {
                final answer = question["answers"][index];
                final isSelected = selectedIndexes.contains(index);

                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                  child: CustomButton(
                    text: answer["text"],
                    onPressed: () {
                      setState(() {
                        if (selectOption == 1) {
                          selectedIndexes = [index];
                        } else {
                          if (isSelected) {
                            selectedIndexes.remove(index);
                          } else {
                            if (selectedIndexes.length < selectOption) {
                              selectedIndexes.add(index);
                            }
                          }
                        }

                        final selectedAnswers =
                            selectedIndexes
                                .map<Map<String, int>>(
                                  (i) => {
                                    'questionId': question['id'] as int,
                                    'selectedOption': i + 1,
                                  },
                                )
                                .toList();

                        handleAnswer(selectedAnswers);
                      });
                    },
                    backgroundColor:
                        isSelected
                            ? const Color(0xff2D7FFB)
                            : const Color(0xffD9E8FF),
                    color: isSelected ? Colors.white : const Color(0xff2D7FFB),
                  ),
                );
              }),
              const SizedBox(height: 80),
            ],
          ),
        ),
      ),
      bottomNavigationBar: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        decoration: BoxDecoration(
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 8,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        child: Row(
          children: [
            Expanded(
              child: SizedBox(
                height: 47,
                child: OutlinedButton(
                  onPressed: () {
                    if (widget.questionIndex == 0) {
                      widget.onExitSurvey();
                    } else {
                      widget.backPressed();
                    }
                    setState(() {
                      selectedIndexes = [];
                    });
                  },
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Color(0xFFD1D5DB), width: 1),
                    backgroundColor: Colors.white,
                    foregroundColor: const Color(0xff374151),
                    textStyle: const TextStyle(fontWeight: FontWeight.w500),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: const Text("이전"),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: SizedBox(
                height: 47,
                child: CustomButton(
                  text: "다음",
                  onPressed:
                      selectedIndexes.isEmpty
                          ? () {}
                          : () {
                            final selectedAnswers =
                                selectedIndexes
                                    .map<Map<String, int>>(
                                      (i) => {
                                        'questionId': question['id'] as int,
                                        'selectedOption': i + 1,
                                      },
                                    )
                                    .toList();

                            handleAnswer(selectedAnswers);
                            handleNext();
                          },
                  backgroundColor:
                      selectedIndexes.isEmpty
                          ? Colors.grey.shade300
                          : const Color(0xffD9E8FF),
                  color:
                      selectedIndexes.isEmpty
                          ? Colors.grey
                          : const Color(0xff3578FF),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
