from setuptools import setup, find_packages

setup(
    name='pip_detector',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'click',
        'opencv-python',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'pip_detector = pip_detector.cli:main',
        ]
    }
)
