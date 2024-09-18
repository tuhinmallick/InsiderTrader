import sys
import requests

def check_version(package, version):
    url = f"https://pypi.org/pypi/{package}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if version in data['releases']:
            return True
    return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_version.py <package> <version>")
        sys.exit(1)
    package = sys.argv[1]
    version = sys.argv[2]
    exists = check_version(package, version)
    if exists:
        print(f"Version {version} of package {package} already exists on PyPI.")
        sys.exit(1)
    else:
        print(f"Version {version} of package {package} does not exist on PyPI.")
        sys.exit(0)
