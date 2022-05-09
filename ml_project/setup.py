from setuptools import find_packages, setup


REQUIREMENTS_TXT = "requirements.txt"


def req_parse(path: str) -> list:
    with open(path, "r") as fin:
        requirements = []
        for line in fin:
            if line.startswith("-r "):
                _, requirements_path = line.split(" ", 1)
                with open(requirements_path.strip(), "r") as inner_fin:
                    line_requirements = [line.strip() for line in inner_fin]
                requirements.extend(line_requirements)
            else:
                requirements.append(line.strip())
    return requirements


setup(
    name="homework1",
    packages=find_packages(),
    version="0.1.0",
    description="""
        HW01 for ML in Prod 
    """,
    author="Julia Chakvetadze",
    license="MIT",
    install_requires=req_parse(REQUIREMENTS_TXT),
)