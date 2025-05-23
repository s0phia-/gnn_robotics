import subprocess
result = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
print("Active screen sessions:")
print(result.stdout)
