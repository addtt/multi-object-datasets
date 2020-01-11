import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
packages = [p for p in packages if p.startswith('multiobject')]

setuptools.setup(
    name="multiobject",  # Replace with your own username
    version="0.0.1",
    author="Andrea Dittadi",
    author_email="andrea.dittadi@gmail.com",
    description="Tools to generate and use multi-object datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/addtt/multi-object-datasets",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
    ]
)
