from openpyxl import Workbook
import openpyxl
import glob
import re


class XLSXReader:
    def __init__(self):
        pass


if __name__ == "__main__":
    book = openpyxl.load_workbook('../data/sample.xlsx')
    sheet = book.active
    cells = sheet['A1': 'J10']

    r=1
    c=1
    clusters = {}
    for row in cells:
        for col in row:
            color = col.fill.start_color.index
            if color not in clusters:
                clusters[color] = []
            clusters[color].append((r, c))
            c = c+1
            # print(col.value, "\t")
        r = r+1
        c = 1

    for color in clusters:
        print(clusters[color])