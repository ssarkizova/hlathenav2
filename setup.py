

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
        'importlib-resources>=5.12.0',
        'logomaker>=0.8',
        'numpy>=1.20.0',
        'pandas>=1.3.5',
        'scipy>=1.7.3',
        'umap-learn>=0.5.3',
        'scikit-learn>=1.0.2',
        'matplotlib>=3.5.1',
        'seaborn>=0.13.0',
        'torch>=2.1.0',
        'Bio>=1.6.0',
        'pyahocorasick>=2.0.0'], # add req.txt

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True,
        package_data={'': ['data/*', 
                           'data/model_training/*', 
                           'data/motif_entropies/*', 
                           'data/projection_models/*',
                           'references/*',
                           'references/expression/*']},
)


#include_package_data=False,
#package_data={'': ['data/*']},
