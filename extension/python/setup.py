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


from distutils.core import setup, Extension

module1 = Extension('pygemmulem',
#                    define_macros = [('DEBUG', '1')],
#                    include_dirs=['../../src/lib'],
#                    libraries=['stdc++'],
                    libraries=['em'],
#                    extra_objects = ['../../build/src/lib/libem.a'],
                    sources = ['pygemmulem.c'])

setup(name = 'pygemmulem',
        version = '1.0',
        ext_modules = [module1])
