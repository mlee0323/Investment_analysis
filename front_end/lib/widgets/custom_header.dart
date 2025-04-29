import 'package:flutter/material.dart';

class CustomHeader extends StatelessWidget implements PreferredSizeWidget {
  final bool showBackButton;
  final bool showLogo;
  final String? title;
  final bool showUserIcon;
  final VoidCallback? onPressed;

  const CustomHeader({
    super.key,
    this.showBackButton = false,
    this.showLogo = false,
    this.title,
    this.showUserIcon = false,
    this.onPressed,
  });

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);

  @override
  Widget build(BuildContext context) {
    return Material(
      elevation: 1,
      shadowColor: Colors.grey.withAlpha((0.2 * 255).round()),
      color: Colors.white,
      child: SafeArea(
        child: Container(
          height: kToolbarHeight,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              if (showBackButton)
                IconButton(
                  icon: const Icon(Icons.arrow_back, color: Colors.black),
                  onPressed: () => Navigator.pop(context),
                )
              else if (showLogo)
                TextButton(
                  child: Image.asset('images/logo.png', height: 19),
                  onPressed: () {
                    Navigator.pushNamed(context, "/");
                  },
                )
              else
                const SizedBox(width: 24),

              Expanded(
                child: Center(
                  child:
                      title != null
                          ? Text(
                            title!,
                            style: const TextStyle(
                              color: Color(0xff4B5563),
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                            ),
                          )
                          : const SizedBox.shrink(),
                ),
              ),

              if (showUserIcon)
                const CircleAvatar(
                  radius: 20,
                  backgroundColor: Color(0xffD9E8FF),
                  child: Icon(Icons.person, color: Color(0xff4B5563), size: 25),
                )
              else
                const SizedBox(width: 40),
            ],
          ),
        ),
      ),
    );
  }
}
