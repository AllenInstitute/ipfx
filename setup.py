from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('test_requirements.txt','r') as f:
    test_requirements = f.read().splitlines()

setup(
    name = 'aibs_ipfx',
    version = '0.1.0',
    description = """intrinsic physiology feature extractor""",
    author = "David Feng",
    author_email = "davidf@alleninstitute.org",
    url = 'https://github.com/AllenInstitute/aibs.ipfx',
    packages = find_packages(),
    include_package_data=True,
    install_requires = requirements,
    entry_points={
          'console_scripts': [
              'aibs.ipfx = aibs.ipfx.__main__:main'
        ]
    },
    license="Allen Institute Software License",
    setup_requires=['pytest-runner'],
    tests_require = test_requirements
)
