import setuptools

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="vesuvius",
    version="0.0.0",
    author="Branden Chan",
    packages=setuptools.find_packages(),
    install_requires=install_requires
)
