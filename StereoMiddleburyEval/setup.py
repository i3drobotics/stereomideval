import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stereo-mideval",
    version="1.0.1",
    author="Ben Knight",
    author_email="bknight@i3drobotics.com",
    description="Evaluation dataset and tools from Middlebury Stereo Evaulation data 2014.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/i3drobotics/StereoMiddleburyEval",
    packages=setuptools.find_packages(),
    package_dir={'StereoMiddleburyEval':'StereoMiddleburyEval'},
    install_requires=[
        'numpy','opencv-python','wget'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)