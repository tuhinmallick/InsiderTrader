import sys
import requests


def check_version(package, version):
    """
    Retrieves metadata from PyPI and checks if a specified package has released
    the given version. It returns `True` if the version is found, indicating the
    package supports it, and `False` otherwise. The check relies on successful
    HTTP request to retrieve package data.

    Args:
        package (str): Used to specify the name of a Python package for which the
            version will be checked on PyPI (Python Package Index). It represents
            the identifier of the software being queried.
        version (str): Used as a value to search for in the 'releases' list of the
            package's data retrieved from PyPI, indicating a specific version
            number of the package.

    Returns:
        bool: 1 for True and 0 for False, indicating whether a specified package
        version exists on PyPI. The result depends on the package name and version
        provided as arguments.

    """
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
