import 'package:flutter/material.dart';

class Snackbar {
  final String text;
  final IconData icon;
  final int duration;
  final Color color;
  final Color backgroundColor;

  Snackbar({
    required this.text,
    this.icon = Icons.info,
    this.duration = 2,
    this.color = Colors.white,
    this.backgroundColor = Colors.grey,
  });

  void showSnackbar(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        duration: Duration(seconds: duration),
        backgroundColor: Colors.transparent,
        elevation: 0,
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.only(bottom: 20.0, left: 24.0, right: 24.0),
        content: Align(
          alignment: Alignment.center,
          child: Container(
            width: 450,
            padding: const EdgeInsets.symmetric(
              horizontal: 16.0,
              vertical: 12.0,
            ),
            decoration: BoxDecoration(
              color: backgroundColor,
              borderRadius: BorderRadius.circular(8),
            ),

            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: color),
                const SizedBox(width: 8),
                Flexible(
                  child: Text(
                    text,
                    style: TextStyle(color: color),
                    overflow: TextOverflow.ellipsis,
                    maxLines: 2,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
