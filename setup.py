import setuptools


def get_readme():
    with open('README.md') as f:
        return f.read()

INSTALL_REQUIRES = [
    'opencv-contrib-python==4.5.1.48',
    'imutils==0.5.4',
]
TESTS_REQUIRE = ['pytest']


setuptools.setup(
    name='objectdaddy',
    version='0.0.3',
    description='A python yolov3-tiny-based object recognizer.',
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    keywords='rtsp stream python',
    url='http://github.com/ryderdamen/objectdaddy',
    author='Ryder Damen',
    author_email='dev@ryderdamen.com',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    test_suite='pytest',
    tests_require=TESTS_REQUIRE,
    include_package_data=True,
)
