import sys
import os
import traceback

# Add project root to path
sys.path.append(os.getcwd())

print("--- Starting Runtime Import Verification ---")

modules_to_test = [
    "signal_mailer.portfolio_manager",
    "signal_mailer.html_generator",
    "signal_mailer.signal_detector",
    "signal_mailer.index_sniper",
    "signal_mailer.debate_council",
    "signal_mailer.mailer_service",
    "signal_mailer.integrated_run",
]

for mod_name in modules_to_test:
    try:
        print(f"Importing {mod_name}...")
        __import__(mod_name)
        print(f"✅ {mod_name} imported successfully.")
    except Exception as e:
        print(f"❌ FAILED to import {mod_name}")
        traceback.print_exc()
        sys.exit(1)

print("--- All Modules Imported Successfully ---")
