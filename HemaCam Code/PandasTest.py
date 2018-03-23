# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:11:04 2018

@author: eshikasaxena
"""

import pandas as pd
import csv

inpath = '..\\HemaCam-Data\\Segmented_Cells\\Cell_Properties'
outpath = '..\\HemaCam-Data\\Segmented_Cells\\Cell_Properties'
csvInpfile = inpath + '\\Sickle3_data_results.csv'
htmlOutfile = outpath + '\\pandas.html'

df = pd.read_csv(csvInpfile)
file = open(htmlOutfile, 'w')
file.write(df.to_html(justify='center', escape = False))
file.close()




summaryOutfile = outpath + '\\summary.html'
summaryOutCsv = outpath + '\\summary.csv'

#summaryDF = df.groupby('Result').count()


#CODE STARTS HERE        

count = df['Result'].count()
data = df['Result'].value_counts(dropna=True)
regular = data.get('Regular')
sickle = data.get('Sickle')
total_data = [['Regular Cells', str(regular)], ['Sickle Cells', str(sickle)], ['Total Cells', str(count)]]

with open(summaryOutCsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Summary', 'Count'])
        for x in total_data:
            writer.writerow(x)
#        writer.writerow(['Count', 'Regular', 'Sickle'])
#        writer.writerow([count, regular, sickle])


df = pd.read_csv(summaryOutCsv)
file = open(summaryOutfile, 'w')
file.write(df.to_html(justify='center', escape = False))
file.close()
f.close()


#print('END            ')



#summaryHeader = ['Total']
#row1 = ['Total cells',  df['Result'].count()]
#row2 = [df['Result'].value_counts(dropna=False) ]
##row2 = [df['Result'].count() ]
#sf = pd.DataFrame([row2], columns=summaryHeader)
#print(sf)
#print (sf)
#print (row2)

#print (df)

#summaryFile = open(summaryOutfile, 'w')
##                    summaryFile.write(summaryDF.to_html(justify='center', escape = False))

#summaryFile.write(summaryDF.ix[:, 2].to_html(justify='center', escape = False))
#summaryFile.close()


 
#df = pd.read_csv(outpath)
#df.head(8)
#
##columns = ['Cell Image', 'Count', 'Perimeter', 'Area', 'Circularity', 'Major Axis', 'Width', 'Length']
#df = pd.read_csv(csvOutfile)
##df.head(8)
##print (df)
##print (df.to_html())
#
#file = open(htmlOutfile, 'w')
#file.write(df.to_html(escape = False))
#

#from IPython.display import HTML
#df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
## this file is in the same folder as the notebook I'm using on my drive
#df['image'] = 'so-logo.png'
#df['new'] = df['image'].apply(lambda x: '<img src="{}"/>'.format(x) if x else '')
#HTML(df.to_html(escape=False))

# Begin changes
#            features = [ '<img src=' + '"..\\Cell_Images\\' + imgname + "_{}".format(count) + ".png" + '">'] + [count] + features
           
# End changes



#
#def image_formatter(im):
#    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
#
#HTML(dogs[['breed', 'image']].to_html(formatters={'image': image_formatter}, escape=False))


#columns = ['Cell Image', 'Count', 'Perimeter', 'Area', 'Circularity', 'Major Axis', 'Minor Axis', 'Ratio']
#df = pd.read_csv(outpath, names=columns)

# This you can change it to whatever you want to get
# age_15 = df[df['age'] == 'U15']
# Other examples:
# bye = df[df['opp'] == 'Bye']
# crushed_team = df[df['ACscr'] == '0']
# crushed_visitor = df[df['OPPscr'] == '0']
# Play with this

# Use the .to_html() to get your table in html
# print(crushed_visitor.to_html())
#print (df.to_html())
