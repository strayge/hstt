from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='hstt',
    version='0.0.2',
    description='HTTP stress testing tool',
    long_description=readme,
    author='strayge',
    author_email='strayge@gmail.com',
    long_description_content_type="text/markdown",
    url="https://github.com/strayge/hstt",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'hstt = hstt.main:main',
        ],
    },
    python_requires='>=3.7',
    install_requires=requirements,
    data_files=[('', ['LICENSE'])],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
    ],
)
