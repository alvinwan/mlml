from setuptools import setup

tests_require = []
install_requires = []

VERSION = '0.0.1'

setup(
    name="MLML",
    version=VERSION,
    author="Alvin Wan",
    author_email='hi@alvinwan.com',
    description=("offers support for a variety of algorithms on memory-limited "
                 "infrastructure"),
    license="BSD",
    url="https://github.com/alvinwan/mlml",
    packages=['mlml'],
    tests_require=tests_require,
    install_requires=install_requires + tests_require,
    download_url='https://github.com/alvinwan/mlml/archive/%s.zip' % VERSION,
    classifiers=[
        "Topic :: Utilities",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
    ],
)