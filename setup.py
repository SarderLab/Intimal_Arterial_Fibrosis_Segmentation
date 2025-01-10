import importlib
import sys

from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    sys.stderr.write("""scikit-build is required to build from source or run tox.
Please run:
  python -m pip install scikit-build
""")
    # from setuptools import setup
    sys.exit(1)



setup(
    name='Intimal_Arterial_Fibrosis_Segmentation',
    use_scm_version={'local_scheme': 'no-local-version',
                     'fallback_version': '0.0.0'},
    description='A Python plugin for performing Intimal Segmentation',
    long_description_content_type='text/x-rst',
    author='Suhas Katari Chaluva Kumar',
    author_email='skumar@dynanetcorp.com',
    url='https://github.com/SarderLab/Intimal_Arterial_Fibrosis_Segmentation',
    packages=find_packages(exclude=['tests', '*_test*']),
    package_dir={
        'itseg': 'itseg',
    },
    include_package_data=True,
    install_requires=[
        'girder-client',
        'girder-slicer-cli-web',
        # scientific packages
        'matplotlib',
        'pandas',
        'openpyxl',
        'scikit-learn',
        'shapely',
        'opencv-python',
        'tiffslide==2.4.0',
        'torch',
        'scikit-image',
        'torchvision',
        'scipy',
        'zarr==2.18.2',
        # cli
        'ctk-cli',
    ],
    license='Apache Software License 2.0',
    keywords='Intimal Segmentation',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
    python_requires='>=3.9',
)