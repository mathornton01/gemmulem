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
# EM demo

library(rgemmulem)

fname <- system.file("extdata", "em_tst.csv", package="rgemmulem")
data <- read.csv(fname, header=FALSE, colClasses = c("character", "numeric"));


compats <- unlist(data[1], use.names=FALSE)
counts <- as.integer(unlist(data[2], use.names=FALSE))

result <- rgemmulem_expectationmaximization(compats, counts)
