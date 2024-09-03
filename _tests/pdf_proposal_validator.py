import glob

proposal_pdfs = glob.glob("pdfs/*.pdf")

if len(proposal_pdfs) < 1:
    print(f'Too few files. You should have 1 joint project proposal. (You have {len(proposal_pdfs)} in total.)')
    exit()
elif len(proposal_pdfs) > 1:
    print(f'Too many files. You should only have 1 project proposal for your group. (You have {len(proposal_pdfs)} in total.)')
    exit()

print('The project proposal exists.')