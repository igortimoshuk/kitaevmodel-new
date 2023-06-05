from setuptools import setup, find_packages

VERSION = '0.1.2' 
DESCRIPTION = 'A tool for numerical simulation of Kitaev honeycomb model'
LONG_DESCRIPTION = 'A tool for numerical simulation of Kitaev honeycomb model'

setup(
        name="kitaevmodel", 
        version=VERSION,
        author="Igor Timoshuk",
        author_email="<iltimoshuk@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'tensorflow', 
        'networkx', 'matplotlib', 'scipy', 'sympy'],
        
        keywords=['python', 'Kitaev model'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ], 
        test_suite='nose.collector',
        tests_require=['nose']
)