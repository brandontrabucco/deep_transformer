"""Setup script for wordpack."""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy', 'tensorflow-gpu', 'nltk', 'matplotlib']


setup(name='wordpack', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('wordpack')],
    description='Brandon Trabucco Word Pack Repo')