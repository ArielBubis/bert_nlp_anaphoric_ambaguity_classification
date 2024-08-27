import os
import subprocess
import sys
# Check if running inside a virtual environment
def check_venv():
    print("Checking for virtual environment...")
    if sys.prefix == sys.base_prefix:
        print("Virtual environment not detected. Creating one...")
        create_venv()
    else:
        print("Virtual environment detected.")
        install_requirements()

# Create and activate virtual environment
def create_venv():
    venv_path = os.path.join(os.getcwd(), ".venv")
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    print("Virtual environment created successfully.")
    install_requirements()
    # activate_venv(venv_path)

def activate_venv(venv_path):
    activate_script = os.path.join(venv_path, "Scripts", "activate")
    with open(activate_script) as file:
        exec(file.read(), {'__file__': activate_script})
    install_requirements()
    print("Virtual environment activated.")

# Install required packages
def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

# Check virtual environment
check_venv()

