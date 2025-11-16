#!/usr/bin/env python
# Test runner for generate_missing_charts.py

import sys
import os

os.chdir("c:\\Users\\Admin\\Documents\\sustainability_project")

print("Starting chart generation...")
print("=" * 70)

try:
    # Execute the chart generation script
    exec(open("generate_missing_charts.py").read())
    print("\n" + "=" * 70)
    print("SUCCESS: All charts generated!")
    print("=" * 70)
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
