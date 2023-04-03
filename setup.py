from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='mitre_histology_toolkit',
   version='1.0',
   description='A useful module',
   license="MIT",
   long_description=long_description,
   author='Man Foo',
   author_email='foomail@foo.com',
   url="http://www.foopackage.com/",
   packages=['mitre_histology_toolkit'],  #same as name
   install_requires=['pandas', 'numpy']#, #external packages as dependencies
    # scripts=[
    #          'scripts/cool',
    #          'scripts/skype',
    #         ]
)