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

#
# ./setup.py build
# ./setup.py install
#
import pygemmulem
import sys, os
import array
from argparse import ArgumentParser, FileType



def run_coarse_multinomial_mode(fp, verbose=False, maxiter=1000, rtole=0.00001):

    # load compatiblity pattern and count
    #
    comppattern = []
    counts = []

    for line in fp:
        line = line.strip().split(',')
        
        comppattern.append(line[0])
        counts.append(int(line[1]))



    result = pygemmulem.expectationmaximization(comppattern, counts, verbose)

    return result


def run_gaussian_mixture_mode(fp, num_dist=3, verbose=False, maxiter=1000, rtole=0.00001):

    # load values
    values = []

    for line in fp:
        line = line.strip().split(',')
        values.append(float(line[0]))

    ar = array.array('d', values) 
    result = pygemmulem.unmixgaussians(ar, num_dist, verbose=verbose, maxiter=maxiter, rtole=rtole)

    return result


def run_exponential_mixture_mode(fp, num_dist=3, verbose=False, maxiter=1000, rtole=0.00001):

    # load values
    values = []

    for line in fp:
        line = line.strip().split(',')
        values.append(float(line[0]))

    ar = array.array('d', values) 
    result = pygemmulem.unmixexponentials(ar, num_dist, verbose=verbose, maxiter=maxiter, rtole=rtole)

    return result


if __name__ == '__main__':
    parser = ArgumentParser(
            description='GEMMULEM')

    parser.add_argument('input_fp',
            nargs='?',
            type=FileType('r'),
            help='Input data. list of counts with compatibility patters or list of values')


    group = parser.add_mutually_exclusive_group()

    group.add_argument('-i',
            dest='cmm',
            action='store_true',
            default=False,
            help='Run EM in coarse multinomial mode')

    group.add_argument('-g',
            dest='gmm',
            action='store_true',
            default=False,
            help='Run EM in gaussian mixture deconvolution mode')

    group.add_argument('-e',
            dest='emm',
            action='store_true',
            default=False,
            help='Run EM in exponential mixture deconvolution mode')

    parser.add_argument('-n', '--num-dist',
            dest='numdist',
            action='store',
            default=3,
            type=int,
            help='Number of distributions')

    parser.add_argument('--verbose',
            dest='verbose',
            action='store_true',
            default=False,
            help='')


    args = parser.parse_args()

    if not args.input_fp:
        parser.print_help()
        exit(1)
        

    if not args.cmm and not args.gmm and not args.emm:
        parser.print_help()
        exit(2)


    r = None
    if args.cmm:
        r = run_coarse_multinomial_mode(args.input_fp, args.verbose)
    elif args.gmm:
        r = run_gaussian_mixture_mode(args.input_fp, args.numdist, args.verbose)
    elif args.emm:
        r = run_exponential_mixture_mode(args.input_fp, args.numdist, args.verbose)

    print(list(r))

