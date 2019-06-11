# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:45:35 2019

@author: Markus.Meister1
"""

def xlsx(fname,sheet, skip=0, header=0):
    import zipfile
    from xml.etree.ElementTree import iterparse
    import re
    z = zipfile.ZipFile(fname)
    if 'xl/sharedStrings.xml' in z.namelist():
        # Get shared strings
        strings = [element.text for event, element
                   in iterparse(z.open('xl/sharedStrings.xml')) 
                   if element.tag.endswith('}t')]
    sheetdict = { element.attrib['name']:element.attrib['sheetId'] for event,element in iterparse(z.open('xl/workbook.xml'))
                                      if element.tag.endswith('}sheet') }
    rows = []
    row = {}
    value = ''
    n_row = 0
    
    if sheet in sheetdict:
        sheetfile = 'xl/worksheets/sheet'+sheetdict[sheet]+'.xml'
        #print(sheet,sheetfile)
        for event, element in iterparse(z.open(sheetfile)):
            # get value or index to shared strings
            if element.tag.endswith('}v') or element.tag.endswith('}t'):
                value = element.text
            # If value is a shared string, use value as an index
            if element.tag.endswith('}c'):
                if element.attrib.get('t') == 's':
                    value = strings[int(value)]
                # split the row/col information so that the row leter(s) can be separate
                letter = re.sub('\d','',element.attrib['r'])
                row[letter] = value
                value = ''
            if element.tag.endswith('}row'):
                rows.append(row)
                row = {}

    return rows