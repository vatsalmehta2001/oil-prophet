"""
Simple test script to check if PyEMD can be imported properly.
"""

import os
import sys

# Print Python version and path
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Try importing PyEMD
try:
    from PyEMD import CEEMDAN
    print("✅ Successfully imported PyEMD.CEEMDAN")
    print(f"PyEMD location: {CEEMDAN.__module__}")
except ImportError as e:
    print(f"❌ Failed to import PyEMD: {e}")
    
    # List installed packages
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    print("\nInstalled packages:")
    print(result.stdout)
    
    # Try to provide helpful information
    print("\nTroubleshooting tips:")
    print("1. Install PyEMD: pip install PyEMD")
    print("2. Make sure you're using the correct Python environment")
    print("3. Check if PyEMD is installed for this specific Python interpreter")

# List sys.path to see where Python is looking for modules
print("\nPython module search paths:")
for path in sys.path:
    print(f"  {path}")