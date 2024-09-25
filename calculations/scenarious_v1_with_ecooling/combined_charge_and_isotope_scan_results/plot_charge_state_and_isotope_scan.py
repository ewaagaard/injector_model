"""
Load charge state and isotope scan data, and plot
"""
import json


# Load isotope scan
with open("../isotope_scan_no_PS_limit/output/isotope_scan.json", "r") as fp:
   isotope_dict = json.load(fp)
