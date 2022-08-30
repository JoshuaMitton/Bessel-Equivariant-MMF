from setuptools import setup
from setuptools import find_packages

setup(name='Bessel-Equivariant-MMF',
      version='1.0',
      description='Bessel-Equivariant-MMF in PyTorch',
      author='Josh Mitton',
      author_email='j.mitton.1@research.gla.ac.uk',
      url='https://joshuamitton.github.io',
      download_url='https://github.com/JoshuaMitton/Bessel-Equivariant-MMF',
      license='GNU GENERAL PUBLIC LICENSE',
      install_requires=['numpy>=1.20.3',
                        'torch>=1.10.1+cu113',
                        'torchvision>=0.11.2+cu113',
                        'tqdm>=4.62.3',
                        'matplotlib>=3.5.1',
                        'h5py>=3.6.0',
                        'csv>=1.0',
                        'e2cnn>=0.2.1',
                        'argparse>=1.1'
                        'opencv-python>=4.5.5'],
      package_data={'Bessel-Equivariant-MMF': ['README.md']},
      packages=find_packages())
