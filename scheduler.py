import subprocess
import time


while True:
    print("Running DVC pipeline...")
    subprocess.run(["dvc", "repro"], check=True)

    print("Ensuring Docker services are running...")
    subprocess.run([
        "docker", "compose",
        "-f", "code/deployment/docker-compose.yml",
        "up", "-d"
    ], check=True)

    print("Next run in 5 minutes.\n")
    time.sleep(300)