#!/usr/bin/env python
"""Execute verify_and_generate_charts.py with proper error handling"""

import subprocess
import sys
import os

os.chdir("c:\\Users\\Admin\\Documents\\sustainability_project")

print("\n" + "=" * 80)
print("EXECUTING CHART VERIFICATION AND GENERATION")
print("=" * 80 + "\n")

try:
    # Execute the verification script
    result = subprocess.run(
        [sys.executable, "verify_and_generate_charts.py"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    sys.exit(result.returncode)
    
except subprocess.TimeoutExpired:
    print("ERROR: Script execution timed out after 120 seconds")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
