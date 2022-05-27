from setuptools import setup, find_packages

setup(
    name="markov-abstractions",
    author="Gabriel Paludo Licks, Marco Favorito",
    author_email="licks@diag.uniroma1.it, favorito@diag.uniroma1.it",
    version="0.1.0",
    url="https://github.com/whitemech/markov_abstractions.git",
    packages=find_packages(where="markov_abstractions"),
    package_dir={"": "markov_abstractions"},
    include_package_data=True,
    zip_safe=False,
    install_requires=["gym", "graphviz"],
    license="GPL-3.0-or-later",
    keywords=[
        "reinforcement learning",
        "non-markovianity",
        "markov abstraction",
        "illusion module",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite='tests',
)
