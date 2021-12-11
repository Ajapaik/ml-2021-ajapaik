from setuptools import setup, find_packages
setuptools.setup( 
    name='object_detector', 
    version='0.1.0', 
    packages=find_packages(where='src'), 
    package_dir={"": "src"},
    entry_points={ 
        'console_scripts': [ 
            'object_detector = object_detector:main' 
        ] 
    }, 
    install_requires=[
        'opencv-python',
        'numpy'
    ],
    
)
