#!/usr/bin/env python3

# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
  Utility for renumbering code cells in a notebook.
"""

import os.path
import argparse
import sys
import re
import json

# global value starts at 1 for the first code block
cell_number = 1

# precompile a useful regex that matches an old '# cell 234' line of code
has_cell_number = re.compile(r'^\s*#\s*cell\s*#?\s*\d+\s*$', re.IGNORECASE)

# abort with error message if the commandline arg is not correct
def arg_check(path): 
  if not isinstance(path,str):
    print('The notebook specified is not a string')
    sys.exit()

  if not path.endswith('.ipynb'):
    print('The notebook does not have the expected .inbpy suffix')
    sys.exit()

  if not os.path.isfile(path):
    print('The notebook specified not found in filesystem.')
    sys.exit()

# return a renumbered cell of code
def renumber_code_cell(code_cell):
  global cell_number

  # strip any old numbers
  code_cell["source"] = [ line_of_code for line_of_code in code_cell["source"] if not has_cell_number.match(line_of_code)]

  # add a new  number
  insert_at_line_number=0 # in general, add to the beginning of the block
  if len(code_cell["source"])>0 and code_cell["source"][0].startswith('%%sh'): 
    # this is rare case where line number cannot be prepended to the block
    insert_at_line_number=1
  code_cell["source"].insert(insert_at_line_number,'# cell {:02d}\n'.format(cell_number))
  cell_number += 1
  return code_cell

# return a renumbered cells in general, (for non-code cells, this is a no-op)
def renumber(cell): 
  if cell["cell_type"] == "code":
    return renumber_code_cell(cell)
  else: 
    return cell

# parse 
def parse(path):
    with open(path,'r',encoding='utf-8') as f:
      data = json.load(f)
    data['cells'] = [ renumber(x) for x in data['cells']]
    with open(path,'w',encoding='utf-8') as f:
      json.dump(data,fp=f,indent=1,ensure_ascii=False) # indent=1 is surprising, but that seems to be default.
      f.write('\n')
    return

# parse commandline arguments
def main():
  global cell_number
  parser = argparse.ArgumentParser(description='Renumbers code cells in notebooks', epilog='`%(prog)s *.ipynb` recommended before commit')
  parser.add_argument('notebook',type=str,nargs="+",help='Notebook file (with .ipynb suffix) with code cells')
  args = parser.parse_args()
  for n in args.notebook: 
    arg_check(n)
    cell_number = 1
    parse(n)

if __name__ == "__main__":
    main()