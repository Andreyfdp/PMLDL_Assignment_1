import subprocess
import time


while True:
    print("Running DVC pipeline...")
    subprocess.run(["dvc", "repro"])
    print("Next run in 5 minutes.\n")
    time.sleep(300)