## Matrix Decompositions in Data Analysis
## Spring semester 2021
##
## Assignment 1
## Due 2 February 2021 at 23:59

## Student's information
## Name:
## Email:
## Student ID #:

##
## This file contains stubs of code to help you to do your 
## assignment. You can fill your parts at the indicated positions
## and return this file as a part of your solution. Remember that
## if you define any functions etc. in other files, you must include 
## those files, as well. You do not have to include standard libraries
## or any files that we provide.
##
## Remember to fill your name and matriculation number above.
##

## Preamble
###########

## You can run this script in R with
#source("assignment1.R")
## but it will not run as it is missing the matrix D from Task 3.

## We use library rworldmap to create pretty maps
## If you don't have it, you have to uncomment the following line
#install.packages("rworldmap")
## If you have issues with rworldmap library, you can omit it (see below)
library(rworldmap)

## Same thing for plotrix
#install.packages("plotrix")
library(plotrix)

## utils.R contains some helper functions
## This might require
#install.packages("clue")
source("utils.R")

## Task 1
##########

## Load the data matrix and coordinates
data <- as.matrix( read.csv("worldclim.csv") )
coord <- read.csv("coordinates.csv")

## Familiarize yourself with the data here e.g.
#colnames(data)
## You should not include this part to your report. 

## Compute the SVD and extract the matrices
SVD = svd(data)
U = SVD$u
S = SVD$d
V = SVD$v

## Prepare for plotting: the extremes of the map
xLim <- c(min(coord["lon"]), max(coord["lon"]))
yLim <- c(min(coord["lat"]), max(coord["lat"]))

## Get and plot a map
## If you couldn't get rworldmap working, ignore these lines
map <- getMap(resolution="low")
plot(map, xlim=xLim, ylim=yLim, asp=1)

## Plot the first column of U. The color indicates the value, with
## red being low, green being middle, and blue being high
## Try
#?color.scale
## for more information about the color scale.
## If you don't have working rworldmap, replace 'points' with 'plot'
points(coord[,1], coord[,2], col=color.scale(U[,1], c(1, 0, 0), c(0, 1, 0), c(0, 0, 1), color.spec="rgb"))

## Alternative plot with different color scheme and filled circles
points(coord[,1], coord[,2], col=color.scale(U[,1], c(0,1), 0.8, 1, color.spec="hsv"), cex=.6, pch=19)

## A color legend to explain the colors
color.legend(xLim[1]+1, yLim[1]-5, xLim[2]-1, yLim[1]-3, c(round(min(U[,1]), 4), round(mean(U[,1]), 4), round(max(U[,1]), 4)), color.scale(sort(U[,1]), c(0,1), 0.8, 1, color.spec="hsv"), gradient="x")

## Plot the second column
plot(map, xlim=xLim, ylim=yLim, asp=1)
points(coord[,1], coord[,2], col=color.scale(U[,2], c(0,1), 0.8, 1, color.spec="hsv"))
color.legend(xLim[1]+1, yLim[1]-5, xLim[2]-1, yLim[1]-3, c(round(min(U[,2]), 4), round(mean(U[,2]), 4), round(max(U[,2]), 4)), color.scale(sort(U[,2]), c(0,1), 0.8, 1, color.spec="hsv"), gradient="x")


## YOUR PART STARTS HERE


## YOUR PART ENDS HERE

#############
## Task 2
#############

## You have to implement the rank selection techniques yourself. 
## Remember to use the normalized data.

## If S contains the singular values, you can plot them by
plot(S, type="l")

## YOUR PART STARTS HERE


## YOUR PART ENDS HERE

#############
## Task 3
#############

## We again use the normalized data. If that is contained in matrix D
## we can compute the k-means to 5 clusters with 10 re-starts as
climate.clustering <- kmeans(D, 5, iter.max=100, nstart=10)$cluster

## The plotting is as in Task 1 (we assume map and xLim and yLim are 
## ready)
plot(map, xlim=xLim, ylim=yLim, asp=1)
points(coord[,1], coord[,2], col=climate.clustering)

## If we want to compare two clusterings, we should match their labels.
## The utils.R file contains match() function to align the labels of
## two clusterings. Here's how to use it

## Open a new device window, e.g. (in Linux)
#X11()

## Compute a new clustering with just one repetition
#other.clustering <- kmeans(D, 5, iter.max=100, nstart=1)$cluster

## Plot the clustering
#plot(map, xlim=xLim, ylim=yLim, asp=1)
#points(coord[,1], coord[,2], col=other.clustering)

## Match the labels
#oc.matched = match(other.clustering, to=climate.clustering)

## Plot the matched labels
#plot(map, xlim=xLim, ylim=yLim, asp=1)
#points(coord[,1], coord[,2], col=oc.matched)


## YOUR PART STARTS HERE


## YOUR PART ENDS HERE
