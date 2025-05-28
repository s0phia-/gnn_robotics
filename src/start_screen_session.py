#!/usr/bin/env python3
import subprocess
import os

print("Starting screen session for main.py...")

# Change to project directory
os.chdir('/home/sop/Projects/gnn_robotics')

# Kill any existing pycharm_session
try:
    result = subprocess.run(['screen', '-X', '-S', 'pycharm_session', 'quit'],
                           capture_output=True, text=True)
except Exception as e:
    print(f"Error killing session: {e}")

# Start new session with your main script
cmd = [
    'screen', '-dmS', 'pycharm_session', 'bash', '-c',
    'cd /home/sop/Projects/gnn_robotics && PYTHONPATH=/home/sop/Projects/gnn_robotics /home/sop/.virtualenvs/gnn_robotics/bin/python src/main.py'
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Screen session 'pycharm_session' started successfully")
    else:
        print(f"Error starting screen session: {result.stderr}")
except Exception as e:
    print(f"Exception starting screen: {e}")

# Verify the session was created
try:
    result = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
    print("Active sessions:")
    print(result.stdout)
except Exception as e:
    print(f"Error listing sessions: {e}")