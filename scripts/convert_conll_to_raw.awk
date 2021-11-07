#! /usr/bin/awk -f
# Author: Jacob Louis Hoover
#
# simple awk script to take a conll file and return plain text, one sentence per line.
# to use: `./convert_conll_to_raw.awk INPUT.conll > OUTPUT.txt`
# note: this does put a space after last word.
{
	if ($1 != "#") {
		if ($0=="") { print"" }
		else { printf "%s ", $2 }
	}
}
END { print "" }
