from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)-> List[str]:
    '''
    Read the requirements file and return the list of requirements
    ''' 
    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='end to end ml project',
    version='0.0.1',
    author='Harshit Aggarwal',
    author_email='harshit0414@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)