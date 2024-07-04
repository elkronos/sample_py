from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sample_py',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'shapely>=1.7.0',
        'pytest>=6.0.0'
    ],  # List of dependencies
    entry_points={
        'console_scripts': [
            
        ],
    },
    author='J',
    author_email='jchase.msu@gmail.com',
    description='A project for various sampling methods.',
    long_description=long_description, 
    long_description_content_type='text/markdown',  # Type of the long description content
    license='MIT', 
    url='https://github.com/elkronos/sample_py',  # URL of your project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],  # Additional classifiers
    python_requires='>=3.6',
)
