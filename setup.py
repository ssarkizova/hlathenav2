

from setuptools import find_packages, setup

setup(
    name='hlathena',
    version='0.1.0',
    description='Immunopeptidomics analyses and HLA binding prediction',
    url='https://github.com/ssarkizova/hlathenav2',
    author='Sisi Sarkizova, Cleo Forman, Mehdi Borji',
    author_email='mehdi.borji.86@gmail.com',
    license='MIT',
    packages=['hlathena'],
    #package_dir={'hlathena':'src'},
    install_requires=[
        'logomaker>=0.8',
        'numpy>=1.20.0',
        'pandas>=1.3.5',
        'scipy>=1.7.3',
        'umap-learn>=0.5.3',
        'scikit-learn>=1.0.2',
        'matplotlib>=3.5.1'], # add req.txt

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True,
    package_data={'': ['data/*']},
)


#include_package_data=False,
#package_data={'': ['data/*']},
