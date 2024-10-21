#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    requirements = []
    dependency_links = []

    for line in lines:
        line = line.strip()
        if line.startswith('-f'):
            dependency_links.append(line.replace('-f ', ''))
        elif line and not line.startswith('#'):
            requirements.append(line)

    return requirements, dependency_links

requirements, dependency_links = parse_requirements('requirements.txt')

setup(
    name="hyperagent",
    version="0.0.1",
    packages=find_packages(exclude=["scripts", "experiments", "atari_results"]),
    install_requires=requirements,
    dependency_links=dependency_links,
)
