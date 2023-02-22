# VALORANT_scoreboard_reader
Python tool using tesseract to OCR a screenshot of a valorant end game scoreboard and turn it into a csv file.

This repository containts two files. 

1. File with the class and all of the functions (scoreboard_reader.py)
2. File for running the OCR tool to gain a csv file (table_to_ocr.py)


## How to use
The first thing you need is a screenshot of a scoreboard like this:
![ss_1](https://user-images.githubusercontent.com/57774007/220695198-47f6b995-b1e4-4fc8-83f6-46325065e388.png)
The tool has been tested on Enlish and Turkish language screenshots.
The tool only works on screenshots in 16:9 aspect ratio currently.

The script will initially ask for a file name i.e. screenshot.png.
This can be modified in the run script if you wish to hardcode the name in.

The tool then splits the image by row and then each row by cell.
Each cell is then ran through tesseract and the converted to a string which can be output.

The output is as a csv file. The script will prompt if you want the file in EU (; as delimiter) or UK (, as delimiter) format.
It is possible to hardcode this into your personal version by altering the write_csv function.
