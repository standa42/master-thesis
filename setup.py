import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='master-thesis-implementation',
    version='0.1a',
    author="Roman Staněk",
    author_email="rnsk@seznam.cz",
    description="Package made for master thesis - mainly for using import features in this project folder structure",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
