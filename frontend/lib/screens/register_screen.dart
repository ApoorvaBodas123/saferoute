import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import 'main_navigation_screen.dart';
import 'login_screen.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;

  void _register() async {
    setState(() {
      _isLoading = true;
    });

    final success = await AuthService.register(
      _nameController.text.trim(),
      _emailController.text.trim(),
      _passwordController.text.trim(),
    );

    setState(() {
      _isLoading = false;
    });

    if (success && mounted) {
      // Remove all previous routes and go to Main Navigation
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(builder: (_) => const MainNavigationScreen()),
        (route) => false,
      );
    } else if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Registration failed. Please fill all fields.'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFD6296),
      body: SingleChildScrollView(
        child: SizedBox(
           height: MediaQuery.of(context).size.height,
           child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Top curved design 
              Expanded(
                flex: 3,
                child: CustomPaint(
                  painter: TopWavePainter(),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.only(left: 32, bottom: 20),
                    alignment: Alignment.bottomLeft,
                    child: const Text(
                      'Register',
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
                flex: 7,
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Full Name', style: TextStyle(color: Colors.white)),
                      const SizedBox(height: 8),
                      _buildTextField(_nameController, obscureText: false),
                      const SizedBox(height: 16),
                      
                      const Text('Email', style: TextStyle(color: Colors.white)),
                      const SizedBox(height: 8),
                      _buildTextField(_emailController, obscureText: false),
                      const SizedBox(height: 16),
                      
                      const Text('Password', style: TextStyle(color: Colors.white)),
                      const SizedBox(height: 8),
                      _buildTextField(_passwordController, obscureText: true),
                      const SizedBox(height: 24),
                      
                      // Social Login Row (Optional here based on mockup)
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
                      // Bottom Row
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          TextButton(
                            onPressed: () {
                               Navigator.of(context).pop(); // Go back to login
                            },
                            child: const Text('Already have an Account? Login', style: TextStyle(color: Colors.white)),
                          ),
                          _isLoading 
                            ? const CircularProgressIndicator(color: Colors.white)
                            : ElevatedButton(
                              onPressed: _register,
                              style: ElevatedButton.styleFrom(
                                foregroundColor: const Color(0xFFFD6296), backgroundColor: Colors.white,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                                padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                              ),
                              child: const Text('Register', style: TextStyle(fontWeight: FontWeight.bold)),
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

  Widget _buildTextField(TextEditingController controller, {bool obscureText = false}) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.2), // Transparent white
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white, width: 1),
      ),
      child: TextField(
        controller: controller,
        obscureText: obscureText,
        style: const TextStyle(color: Colors.white),
        decoration: const InputDecoration(
          border: InputBorder.none,
          contentPadding: EdgeInsets.symmetric(horizontal: 16),
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
