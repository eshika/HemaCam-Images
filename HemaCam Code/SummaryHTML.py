# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 07:59:46 2018

@author: eshikasaxena
"""
import pandas as pd
import csv
import os

def summaryHTML():
    path_save = '../HemaCam-Data/Segmented_Cells/Cell_Properties/Results/'
    csv_files = os.listdir(path_save)
    print(csv_files)
    
    for path in csv_files:
            print(path)
            if '.csv' in path:
                    csv_file = os.path.join(path_save, path)
                    print (path, csv_file)
                    imgname, extension = os.path.basename(csv_file).split(".")
            
                    outpath = '..\\HemaCam-Data\\Segmented_Cells\\Cell_Images\\'
                    csvInpfile = path_save + imgname + '.csv'
                    htmlOutfile = outpath + imgname + '.html'
                    
                    print(htmlOutfile, csvInpfile)
                    df = pd.read_csv(csvInpfile)
                    html_file = open(htmlOutfile, 'w')
                    html_file.write(df.to_html(justify='center', escape = False))
                    html_file.close()
        
    # Write Summary file
                    count = df['Result'].count()
                    data = df['Result'].value_counts(dropna=True)
                    regular = data.get('Regular')
                    sickle = data.get('Sickle')
                    total_data = [['Regular Cells', str(regular)], ['Sickle Cells', str(sickle)], ['Total Cells', str(count)]]
        
                    summaryOutfile = outpath + imgname + '_summary.html'
                    summaryOutCsv = path_save + '..\\..\\Cell_Summary\\' + imgname + '_results_summary.csv'
                    
                    with open(summaryOutCsv, 'w') as rf:
                            writer = csv.writer(rf)
                            writer.writerow(['Summary', 'Count'])
                            for x in total_data:
                                writer.writerow(x)
                                
                    df2 = pd.read_csv(summaryOutCsv)
                    rf.close()
                    summaryFile = open(summaryOutfile, 'w')
                    summaryFile.write(df2.to_html(justify='center', escape = False))
                    summaryFile.close()
        
                    print('Evaluation Finished!')
        
            else:
                    print('No model found')
                    

if __name__ == "__main__":
    summaryHTML()
