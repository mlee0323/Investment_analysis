import 'package:flutter/material.dart';
import 'package:front_end/screens/loadmydata/load_data_screen.dart';
import 'package:front_end/widgets/custom_button.dart';

class MydataScreen extends StatefulWidget {
  const MydataScreen({super.key});

  @override
  State<MydataScreen> createState() => _MyDataLoaderPageState();
}

class _MyDataLoaderPageState extends State<MydataScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: Colors.white, elevation: 1),
      resizeToAvoidBottomInset: false,
      backgroundColor: const Color(0xffF7F7F8),
      body: Center(
        child: Container(
          constraints: BoxConstraints(maxWidth: 450, maxHeight: 300),
          decoration: BoxDecoration(
            color: const Color(0xFFFFFFFF),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 80),

            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.download_for_offline,
                  color: Color(0xffD9E8FF),
                  size: 60,
                ),
                const SizedBox(height: 12),
                const Text(
                  "마이데이터 불러오기",
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                const Text(
                  textAlign: TextAlign.center,
                  "마이데이터를 통해\n내 주식 계좌를 통합하여 조회할 수 있어요.",
                  style: TextStyle(fontSize: 15, color: Color(0xff91929F)),
                ),
                const SizedBox(height: 24),
                CustomButton(
                  text: '마이데이터 불러오기',
                  isFullWidth: true,
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const LoadDataScreen()),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
