'''
This script provides tools for writing the data to excel file.

Authors: Saksham Bhutani, Mohamed Elgendi and Carlo Menon
License: MIT
'''

import os
import xlsxwriter

def save2excel(meth, mse, path, techniques):
    '''
    This function saves the data to excel file.

    Parameters:
        meth (str): Method for removing rPPG signal.
        mse (float): Mean Square Error.
        path (str): Path to save the excel file.

    Returns:
        None
    '''
    workbook = xlsxwriter.Workbook(os.path.join(path,f'img.xlsx'))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    worksheet.write(row, col, "Method")
    worksheet.write(row, col + 1, "MSE")

    row += 1
    for i in range(len(meth)):
        worksheet.write(row, col, meth[i])
        worksheet.write(row, col + 1, mse[i])
        row += 1
    workbook.close()