import json
import os
import ipdb
from tqdm import tqdm
from transformers import AutoTokenizer
import pickle
import random
import ipdb
import xlrd
import xlwt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--baseline_name', type=str)
args = vars(parser.parse_args())


def make_evaluation_file(save_name, data):

    wb = xlwt.Workbook()
    sheet = wb.add_sheet('标注数据')
    sheet.col(1).width = 768 * 40
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.height = 15*11
    style.font = font

    alignment = xlwt.Alignment()
    alignment.horz = 0x01
    alignment.vert = 0x01
    style.alignment = alignment
    style.alignment.wrap = 1

    borders = xlwt.Borders()
    borders.top = xlwt.Borders.THIN
    borders.bottom = xlwt.Borders.THIN
    borders.left = xlwt.Borders.THIN
    borders.right = xlwt.Borders.THIN
    style.borders = borders

    # finish initing the sheet, wb, and style
    line_counter = 0
    for idx, session in tqdm(enumerate(data)):

        prefix = session['prefix']
        method_0 = session['method_0'].replace('<|endoftext|>', '<unk>').replace('##', '')
        method_1 = session['method_1'].replace('<|endoftext|>', '<unk>').replace('##', '')
        if '##' in method_0 or '##' in method_1:
            ipdb.set_trace()

        sheet.write(line_counter, 0, f'前缀-{idx+1}', style)
        sheet.write(line_counter, 1, prefix, style)
        sheet.write(line_counter, 2, '哪一个更好', style)
        line_counter += 1

        sheet.write(line_counter, 0, '候选-1', style)
        sheet.write(line_counter, 1, method_0, style)
        line_counter += 1
        sheet.write(line_counter, 0, '候选-2', style)
        sheet.write(line_counter, 1, method_1, style)
        line_counter += 2

    # set row height
    for i in range(line_counter):
        sheet.row(i).height_mismatch = True
        sheet.row(i).height = 40 * 35
    wb.save(save_name)

with open(f'annotation_files/{args["baseline_name"]}/human_annotation.json') as f:
    data = json.load(f)
    make_evaluation_file(f'annotation_files/{args["baseline_name"]}/copyisallyouneed-{args["baseline_name"]}.xls', data)

