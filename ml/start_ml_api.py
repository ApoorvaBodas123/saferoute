#!/usr/bin/env python3
import subprocess
import time
import sys
import os

def start_ml_api():
    """Start ML API server with auto-restart"""
    print("🚀 Starting ML API Server with auto-restart...")
    
    while True:
        try:
            # Start the server
            proc = subprocess.Popen([
                sys.executable, 
                'ml_api_server_fast.py'
            ], cwd=os.path.dirname(os.path.abspath(__file__)))
            
            print(f"✅ ML API Server started (PID: {proc.pid})")
            print("🌐 Running on http://localhost:8000")
            print("🔄 Auto-restart enabled - will restart if server crashes")
            
            # Wait for process to complete
            proc.wait()
            
            print("❌ ML API Server stopped, restarting in 3 seconds...")
            time.sleep(3)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping ML API Server...")
            if 'proc' in locals():
                proc.terminate()
            break
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            time.sleep(3)

if __name__ == "__main__":
    start_ml_api()
