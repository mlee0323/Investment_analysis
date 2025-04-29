import 'package:flutter/material.dart';

class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback onPressed;
  final isFullWidth;
  final color;
  final backgroundColor;

  const CustomButton({
    super.key,
    required this.text,
    required this.onPressed,
    this.isFullWidth,
    this.color = const Color(0xff3578FF),
    this.backgroundColor = const Color(0xffD9E8FF),
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: isFullWidth == true ? double.infinity : null,
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          foregroundColor: color,
          backgroundColor: backgroundColor,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          minimumSize: const Size.fromHeight(50),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        child: Text(text),
      ),
    );
  }
}
