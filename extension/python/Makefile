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

all: lib

lib:
	python ./setup.py build_ext

install:
	python ./setup.py install --user

clean:
	python ./setup.py clean
	rm -rf dist build test *.so *.egg-info *.egg

test:
	python ./gemmulem_example.py -g ../../src/gmm_tst_vals_1.txt

testinstall:
	python ./setup.py install --install-lib ./
