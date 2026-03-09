import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import 'register_screen.dart';
import 'main_navigation_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;

  void _login() async {
    setState(() {
      _isLoading = true;
    });

    final success = await AuthService.login(
      _emailController.text.trim(),
      _passwordController.text.trim(),
    );

    setState(() {
      _isLoading = false;
    });

    if (success && mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const MainNavigationScreen()),
      );
    } else if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Login failed. Please check your credentials.'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFD6296), // Match mockup pink background
      body: SingleChildScrollView(
        child: SizedBox(
           height: MediaQuery.of(context).size.height,
           child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Top curved design (Using a simple container for now, can be replaced with CustomPainter)
              Expanded(
                flex: 4,
                child: CustomPaint(
                  painter: TopWavePainter(),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.only(left: 32, bottom: 40),
                    alignment: Alignment.bottomLeft,
                    child: const Text(
                      'Log In',
                      style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
              ),
              Expanded(
                flex: 6,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Email', style: TextStyle(color: Colors.white)),
                      const SizedBox(height: 8),
                      // Input field
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: TextField(
                          controller: _emailController,
                          decoration: const InputDecoration(
                            border: InputBorder.none,
                            contentPadding: EdgeInsets.symmetric(horizontal: 16),
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Text('Password', style: TextStyle(color: Colors.white)),
                      const SizedBox(height: 8),
                      // Input field
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: TextField(
                          controller: _passwordController,
                          obscureText: true,
                          decoration: const InputDecoration(
                            border: InputBorder.none,
                            contentPadding: EdgeInsets.symmetric(horizontal: 16),
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Align(
                        alignment: Alignment.centerRight,
                        child: TextButton(
                          onPressed: () {},
                          child: const Text('Forgot Password?', style: TextStyle(color: Colors.white70)),
                        ),
                      ),
                      const SizedBox(height: 16),
                      
                      // Social Login Row
                      Row(
                        mainAxisAlignment: MainAxisAlignment.start,
                        children: [
                           _buildSocialIcon(Icons.g_mobiledata),
                           const SizedBox(width: 16),
                           _buildSocialIcon(Icons.facebook),
                           const SizedBox(width: 16),
                           _buildSocialIcon(Icons.apple),
                        ],
                      ),
                      const Spacer(),
                      // Bottom Row (Register Link & Login Button)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          TextButton(
                            onPressed: () {
                               Navigator.of(context).push(MaterialPageRoute(builder: (_) => const RegisterScreen()));
                            },
                            child: const Text('First Time Here? Register', style: TextStyle(color: Colors.white)),
                          ),
                          _isLoading 
                            ? const CircularProgressIndicator(color: Colors.white)
                            : ElevatedButton(
                              onPressed: _login,
                              style: ElevatedButton.styleFrom(
                                foregroundColor: const Color(0xFFFD6296), backgroundColor: Colors.white,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                              ),
                              child: const Text('Log In', style: TextStyle(fontWeight: FontWeight.bold)),
                          ),
                        ],
                      ),
                      const SizedBox(height: 32),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSocialIcon(IconData icon) {
    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Icon(icon, color: Colors.black87),
    );
  }
}

class TopWavePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    var paint1 = Paint()
      ..color = const Color(0xFFFE8AB1) // Lighter pink for top wave
      ..style = PaintingStyle.fill;

    var path1 = Path();
    path1.moveTo(0, 0);
    path1.lineTo(0, size.height * 0.6);
    path1.quadraticBezierTo(
        size.width * 0.25, size.height * 0.4, size.width * 0.5, size.height * 0.6);
    path1.quadraticBezierTo(
        size.width * 0.75, size.height * 0.8, size.width, size.height * 0.5);
    path1.lineTo(size.width, 0);
    path1.close();
    canvas.drawPath(path1, paint1);
    
    // A second darker wave
    var paint2 = Paint()
      ..color = const Color(0xFFFD4F8A)
      ..style = PaintingStyle.fill;
      
    var path2 = Path();
    path2.moveTo(0, 0);
    path2.lineTo(0, size.height * 0.4);
    path2.quadraticBezierTo(
        size.width * 0.3, size.height * 0.6, size.width * 0.6, size.height * 0.4);
    path2.quadraticBezierTo(
        size.width * 0.8, size.height * 0.3, size.width, size.height * 0.45);
    path2.lineTo(size.width, 0);
    path2.close();
    canvas.drawPath(path2, paint2);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}
