# from setup.py we can make our project as a package where we can use this anywhere we want by this package

from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return list of all requirements present in requirements.txt file
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements] 

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'UnderWaterImageEnhancer',
    version = '0.0.1',
    author = 'ranvitha',
    author_email = 'ranvithaparshi07@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)