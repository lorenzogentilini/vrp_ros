from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['vrp_ros'],
    package_dir={'': 'include'}
)

setup(**setup_args)