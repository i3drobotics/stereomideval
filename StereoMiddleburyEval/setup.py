import setuptools
from setuptools import Command

from os.path import abspath, basename, dirname, join, normpath, relpath
from shutil import rmtree
import glob

here = normpath(abspath(dirname(__file__)))

with open("README.md", "r") as fh:
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

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        global here
        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(normpath(join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print('removing %s' % relpath(path))
                rmtree(path)

setuptools.setup(
    name="stereo-mideval",
    version="1.0.3",
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
    cmdclass={
        'clean': CleanCommand,
    },
)