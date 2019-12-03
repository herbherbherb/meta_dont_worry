import numpy as np
import os
import json
import re
import traceback
import xml.etree.ElementTree as ET
import collections
import PyPDF2


def main():
	path = './data'
	for filename in os.listdir(path):
    	print(filename)

if __name__== "__main__":
	main()