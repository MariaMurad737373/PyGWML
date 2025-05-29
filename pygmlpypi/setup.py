from setuptools import setup, find_packages

setup(
    name='pygml',
    version='0.1.0',
    author='Moin Tariq',
    author_email='sci.mointariq@gmail.com',
    description='Geographically Weighted Machine Learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/moin-t/PyGML/pygmlpypi',  
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'xgboost>=1.7.0',
        'scikit-learn>=1.0.0',
        'libpysal>=4.6.0',
        'esda>=2.4.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)

