# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:50:38 2015

@author: jchen
"""

import json
import glob
import os
import tablib

base_dir='/Users/jchen/Documents/SQLite/lastfm_test'

files = glob.glob(base_dir + '/*/*/*/*.json')

def open_json_file(path):
    data = json.load(open(path))
    return data
    
all_data = [open_json_file(f) for f in files]


headers = all_data[0].keys()
headers.remove('tags')
headers.remove('similars')

ds = tablib.Dataset(headers=headers)

for song in all_data:
    row = [song[h] for h in headers]
    ds.append(row)

open('lastfm_test.csv','w').write(ds.csv)
#open('lastfm_songs.xls','w').write(ds.xls)
#open('lastfm_songs.yaml','w').write(ds.yaml)
