#!/bin/bash
for player in "RandomExpectimaxAgent" "DirectionalExpectimaxAgent"; do
	for ghost in "RandomGhost" "DirectionalGhost"; do
		echo Starting $player with $ghost
		python pacman.py -p $player -l trickyClassic -q -n 5 -a depth=4 -g $ghost
	done
done