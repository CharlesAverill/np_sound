import setuptools

version = '0.0.5'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='np-sound',
    version=version,
    # entry_points={"console_scripts": ["satyrn = satyrn_python.entry_point:main"]},
    author="Charles Averill",
    author_email="charlesaverill20@gmail.com",
    description="A NumPy-based sound library",
    long_description=long_description,
    install_requires=['matplotlib', 'numpy', 'scipy'],
    long_description_content_type="text/markdown",
    url="https://github.com/CharlesAverill/np_sound/",
    packages=['np_sound'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
