"""Setup module to build stereomideval module"""
from os.path import abspath, dirname, join, normpath, relpath
from shutil import rmtree
import glob
import setuptools
from setuptools import Command

with open("../README.md", "r") as fh:
    long_description = fh.read()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info ./__pycache__'.split(' ')

    # Support the "all" option. Setuptools expects it in some situations.
    user_options = [
        ('all', 'a',
         "provided for compatibility, has no extra functionality")
    ]

    boolean_options = ['all']

    def __init__(self,dist):
        self.all = None
        super().__init__(dist)

    def initialize_options(self):
        """Inital options for clean command"""
        self.all = None

    def finalize_options(self):
        """Finalise options for clean command"""

    def run(self):
        """Run clean command"""
        script_path = normpath(abspath(dirname(__file__)))
        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(normpath(join(script_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(script_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, script_path))
                print('removing %s' % relpath(path))
                rmtree(path)

setuptools.setup(
    name="stereo-mideval",
    version="1.0.13",
    author="Ben Knight",
    author_email="bknight@i3drobotics.com",
    description="Evaluation dataset and tools from Middlebury Stereo Evaulation data 2014.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/i3drobotics/stereomideval",
    packages=setuptools.find_packages(),
    package_dir={'stereomideval':'stereomideval'},
    install_requires=[
        'numpy','opencv-python','wget'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    cmdclass={
        'clean': CleanCommand,
    }
)
