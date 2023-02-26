# VALORANT_scoreboard_reader
Python tool using tesseract to OCR a screenshot of a valorant end game scoreboard and turn it into a csv file.

Written by Alex 'Aplox' Porter (twitter.com/_Aplox). Huge thanks to Vladk0r (twitter.com/vladk0r_vlr) for pushing me to finally do this!

This repository containts two files. 

1. File with the class and all of the functions (scoreboard_reader.py)
2. File for running the OCR tool to gain a csv file (table_to_ocr.py)


## How to use
The first thing you need is a screenshot of a scoreboard like this:
![ss_1](https://user-images.githubusercontent.com/57774007/220695198-47f6b995-b1e4-4fc8-83f6-46325065e388.png)
The tool has been tested on English and Turkish language screenshots.
The tool only works on screenshots in 16:9 aspect ratio currently. Unfortunately more testing is required to work with stretched resolution screenshots.

The script will initially ask for a file name i.e. screenshot.png.
This can be modified in the run script if you wish to hardcode the name in.

The tool then splits the image by row and then each row by cell.
Each cell is then passed through tesseract and converted to a string which can be output.

The output is as a csv file. The script will prompt if you want the file in EU (; as delimiter) or UK (, as delimiter) format.
It is possible to hardcode this into your personal version by altering the write_csv function.

The output should look something like this: <br>
![image](https://user-images.githubusercontent.com/57774007/220700904-34984cfc-61cd-4004-b12f-9393d50e6664.png)<br>
The output is sorted alphabetically by name such that all of your team with the same tag should be grouped together.

## Prerequsites
Firstly you need tesseract-OCR. Instructions to install here: <br>
https://medium.com/@ahmedbr/how-to-implement-pytesseract-properly-d6e2c2bc6dda <br>

python dependencies:

sys, opencv, numpy, pytessetact, math, csv, tqdm and PIL.

All dependancies may be installed using the following command:
<code> pip3 install -r "requirements.txt" </code>

## FAQ
Coming soon?
