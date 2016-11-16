"""
Code Submission Verification for MLP projects
Viktor Wegmayr
vwegmayr@inf.ethz.ch
"""

import sys
import os
import zipfile
from shutil import rmtree

PROJECT = 'MLP2'
FINAL_SUB_LENGTH = 139
MIN_DESCRIPTION_LENGTH = 50
ERROR = 0

if len(sys.argv) < 2:
	print("ERROR: No input file specified, usage: 'python checkSub.py PathToYourSubmissionArchive'")
sub = sys.argv[1]

PYTHON3 = 0
if sys.version_info >= (3,0):
	PYTHON3 = 1

WIN = 0
if os.name == 'nt':
	WIN = 1

print("===============================")

# Check if provided submission path exisits

if not os.path.exists(sub):
	print( "FATAL ERROR: Provided submission path does not exist, no such file")
	sys.exit(-1)

# Check archive name

if len(sub.split("/")) > 1:
	sub = sub.split("/")[-1]

if sub.split(".")[-1] != "zip":
	print( "FATAL ERROR: Invalid archive type. Archive needs to be .zip")
	ERROR += 1
	sys.exit(-1)

if len(sub.split("_")) == 1:
	print( "ERROR: Invalid archive name. Archive name needs to be '"+PROJECT+"_teamname.zip'")
	ERROR += 1

if sub.split("_")[0] != PROJECT:
	print( "ERROR: Invalid archive name. Archive name needs to be '"+PROJECT+"_teamname.zip'")
	ERROR += 1

# Check archive structure and content

with zipfile.ZipFile(sub) as file:
	file.extractall(path="tmp")

subfiles = os.listdir("tmp")

if len(subfiles) != 4:
	print( "ERROR: Invalid number of submitted items. There need to be exactly three files and one folder in your archive.")
	ERROR += 1
	if len(subfiles) == 1 and os.path.isdir(subfiles[0]):
		print( "ERROR: Invalid archive structure. Don't put the submission items in an extra folder before zipping.")
		ERROR += 1

found = {'readme' : 0, 'src' : 0, 'predict_final' : 0, 'final_sub.csv' : 0}

for subfile in subfiles:
	if subfile == "readme":
		found['readme'] = 1
	if subfile == "predict_final.py" or subfile == "predict_final.m":
		found['predict_final'] = 1
	if subfile == "final_sub.csv":
		found['final_sub.csv'] = 1
	if subfile == "src":
		found['src'] = 1

for key in found:
	if found[key] == 0:
		if key == "predict_final":
			print( "ERROR: Missing file {}.py (or .m), check spelling.".format(key))
			ERROR += 1
		elif key != "src":
			print( "ERROR: Missing file {}, check spelling.".format(key))
			ERROR += 1
		else:
			print( "ERROR: Missing folder {}, check spelling.".format(key))
			ERROR += 1

# Check src folder
if found['src'] == 1:
	srcFiles = os.listdir("tmp/src")
	if len(srcFiles) == 0:
		print( "WARNING: No files in folder src")

# Check final_sub.csv
if found['final_sub.csv'] == 1:
	#finalSub = np.loadtxt("tmp/final_sub.csv",dtype=str,delimiter=",")
	lines = open("tmp/final_sub.csv")

	finalSub = []
	for line in lines:
		finalSub.append(line)

	if finalSub[0] != "ID,Prediction\n":
		print( "ERROR: Invalid first row in final.csv, first row must be 'ID,Prediction'")
		ERROR += 1

	if len(finalSub) != FINAL_SUB_LENGTH:
		print( "ERROR: Invalid number of rows in final_sub.csv, "+str(FINAL_SUB_LENGTH)+" rows required: 1 header row plus "+str(FINAL_SUB_LENGTH-1)+" id/prediction paris")
		ERROR += 1

	if len(finalSub[0].split(",")) != 2:
		print( "ERROR: Invalid number of columns in final_sub.csv, 2 columns required: ID and Prediction")
		ERROR += 1

# Check predict_final.py
if found['predict_final'] == 1:
	for ext in ['.py','.m']:
		if os.path.exists("tmp/predict_final"+ext):
			fileSize = os.stat("tmp/predict_final"+ext).st_size
			if fileSize == 0:
				print( "ERROR: Empty file: predict_final"+ext)
				ERROR += 1

# Check readme
if found['readme'] == 1:
	lines = []
	with open("tmp/readme") as readme:
		for line in readme:
			if line[0] != "#":
				lines.append(line)

	validEmailCount = 0
	for line in lines:
		if len(line.split("@")) > 2:
			print( "ERROR: More than one email per line in readme, put each email in a separate line")
			ERROR += 1
		elif line.find("@") != -1:
			if line.split(".")[-2] == "ethz":
				validEmailCount += 1
			else:
				print( "ERROR: Invalid email in readme, provide .ethz.ch mail address, each in a separate line")
				ERROR += 1

	if validEmailCount == 0:
		print( "ERROR: No valid author email in readme, insert author .ethz.ch emails in separate lines")
		ERROR += 1

	foundSection = {"Preprocessing" : 0, "Features" : 0, "Model" : 0, "Description" : 0}
	n=1
	for line in lines:
		if line[:-1] == "Preprocessing":
			foundSection["Preprocessing"] = n
		if line[:-1] == "Features":
			foundSection["Features"] = n
		if line[:-1] == "Model":
			foundSection["Model"] = n
		if line[:-1] == "Description":
			foundSection["Description"] = n
		n+=1

	for key in foundSection:
		if foundSection[key] == 0:
			print( "ERROR: Missing Section in readme: "+key+", include this headline")
			ERROR += 1

	if PYTHON3 == 1:
		foundSectionIter = foundSection.items()
	else:
		foundSectionIter = foundSection.iteritems()

	for key, val in foundSectionIter:
		if len(lines[val].split(",")) < 3 and key != "Description" and foundSection[key] > 0:
			print( "ERROR: Not enough keys in section "+key+", at least three comma separated keys required")
			ERROR += 1

	if foundSection["Description"] > 0:
		description = ""
		for line in lines[foundSection["Description"]:]:
			description += line

		if len(description.split(" ")) < MIN_DESCRIPTION_LENGTH:
			print( "ERROR: Insufficient Description in readme, write at least "+str(MIN_DESCRIPTION_LENGTH)+" meaningful words in the description section")
			ERROR += 1

# Print( whether submission passed or failed tests)
if ERROR > 0:
	if WIN == 0:
		print( "\n\033[91mFAIL\033[0m")
	else:
		print( "\n*** FAIL ***")
else:
	if WIN == 0:
		print( "\n\033[92mPASS\033[0m")
	else:
		print( "\n*** PASS ***")

print("===============================")

# Delete tmp

rmtree("tmp")


# Return ERROR sum

sys.exit(ERROR)
# catch ERROR in bash with echo $?
# or just write a python wrapper script