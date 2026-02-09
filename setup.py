from setuptools import setup, find_packages

setup(
    name='ts_sobolev',
    version='0.1',
    packages=find_packages(include=['db_tsw']),
    install_requires=[
        'torch',
    ],
    author='',
    author_email='',
    description='Distance-Based Tree-Sliced Wasserstein Distance',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
