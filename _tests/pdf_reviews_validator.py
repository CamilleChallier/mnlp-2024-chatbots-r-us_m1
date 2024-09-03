import os
import re
import glob

review_pdfs = glob.glob("pdfs/literature reviews/*.pdf")

if len(review_pdfs) < 3:
    print(f'Too few files. You should have 1 paper review per team member. (You have {len(review_pdfs)} in total.)')
    exit()
elif len(review_pdfs) > 4:
    print(f'Too many files. You should only have 1 paper review per team member. (You have {len(review_pdfs)} in total.)')
    exit()

rx = re.compile(r'^\d{6}\.pdf$', re.I)

for file in review_pdfs:
    if not rx.match(os.path.basename(file)):
        print(f'File {file} is not a SCIPER.')
        exit()

print('All paper reviews exist and are named correctly.')