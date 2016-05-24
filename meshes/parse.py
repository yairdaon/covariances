#!/usr/bin/python
import sys
import os

source =  sys.argv[1] + "backup.xml"
    
target = sys.argv[1] + ".xml" 
try:
    os.remove( target )
except:
    pass
target_file = open( target, "a" ) 


with open( source ) as f:
    for line in f:
        if "vertex" in line:
            list_line = line.split('\"')
            list_line[3] = str( float( list_line[3] ) / 1000 )
            list_line[5] = str( float( list_line[5] ) / 1000 )
            curr_line = ""
            for x in list_line:
                curr_line = curr_line + x + '\"' 
                
            target_file.write(curr_line[0:-1])
        else:
            target_file.write(line)

