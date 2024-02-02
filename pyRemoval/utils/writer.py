'''
This script provides tools for writing the data to excel file.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import os
import xlsxwriter

def save2excel(meth, path, metric, metric_name):
    '''
    This function saves the data to excel file.

    Parameters:
        meth (str): Method for removing rPPG signal.
        path (str): Path to save the excel file.
        metric (float): Metric value.
        metric_name (str): Name of the metric.

    Returns:
        None
    '''
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(os.path.join(path,f'img.xlsx'))
    worksheet = workbook.add_worksheet()

    # Add headers
    row = 0
    col = 0
    worksheet.write(row, col, "Method")
    worksheet.write(row, col + 1, metric_name)

    # Write data
    row += 1
    for i in range(len(meth)):
        worksheet.write(row, col, meth[i])
        worksheet.write(row, col + 1, metric[i])
        row += 1
    workbook.close()