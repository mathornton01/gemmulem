#!/usr/bin/python

#
# Copyright 2022, Micah Thornton and Chanhee Park <parkchanhee@gmail.com>
#
# This file is part of GEMMULEM
#
# GEMMULEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GEMMULEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GEMMULEM.  If not, see <http://www.gnu.org/licenses/>.

import sys, os
from setuptools import setup, Extension

EXTRA_INCLUDE_DIR=None
EXTRA_LIB_DIR=None

if 'INCLUDE_DIR' in os.environ:
        EXTRA_INCLUDE_DIR = os.environ['INCLUDE_DIR'].split(';')
        
if 'LIB_DIR' in os.environ:
        EXTRA_LIB_DIR = os.environ['LIB_DIR'].split(';')

module1 = Extension(
        name = 'pygemmulem',
#        define_macros = [('DEBUG', '1')],
        include_dirs = EXTRA_INCLUDE_DIR,
        library_dirs = EXTRA_LIB_DIR,
        libraries=['em'],
        sources = ['pygemmulem.c']
)

setup(
        ext_modules = [module1]
)
