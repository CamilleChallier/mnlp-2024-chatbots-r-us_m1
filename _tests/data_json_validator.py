import glob
import json
import re

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_json_to_dict(filename):
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data

required_general_keys = set(["course_id","question_id", "A_chat_id", "B_chat_id", "A", "B", "ranking_criteria"])
required_ranking_keys = set(["overall", "correctness", "relevance", "clarity", "completeness", "other"])
possible_ranking_values = set(["AB", "A", "B", "None"])

print("Searching for json files...")

filtered_files = glob.glob("data/*.json")
json_cnt = len(filtered_files)

if json_cnt == 0:
    print("ERROR: No json files were found.")
    exit()


has_errors = False

if json_cnt < 3:
    print("#" * 50)
    print(f"ERROR: Too few files. You should have 1 data file per team member. (You only have {json_cnt} in total.)")
    has_errors = True

if json_cnt > 4:
    print("#" * 50)
    print(f"ERROR: Too many files. You should have 1 data file per team member. (You have {json_cnt} in total.)")
    has_errors = True


print(f"Found {json_cnt} json files.")


rx = re.compile(r'^\d{6}\.json$', re.I)

for filepath in filtered_files:

    if not rx.match(os.path.basename(filepath)):
        print("#" * 50)
        print(f'ERROR: File {filepath} is not a SCIPER.')
        has_errors = True

    print("Verifying the format of: ", filepath)
    mydictlist = parse_json_to_dict(filepath)

    dict_cnt = 0
    for mydict in mydictlist:
        ####################################################################
        # Dict keys
        remaining_set = required_general_keys - set(mydict.keys())
        if len(remaining_set) > 0:
            remaining_set_str = ", ".join(remaining_set)
            print("#" * 50)
            print(f"\tERROR: Missing the keys [{remaining_set_str}] in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True
            continue # stop checking this dict because it is missing fields

        ####################################################################
        # Dict values
        # required_general_keys = set(["course_id","question_id", "chat_id", "confidence", "interaction"])
        if type(mydict["course_id"]) != int:
            print("#" * 50)
            print(f"\tERROR: course_id value type is not an int in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True

        if type(mydict["question_id"]) != int:
            print("#" * 50)
            print(f"\tERROR: question_id value type is not an int in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True

        if type(mydict["A_chat_id"]) != int:
            print(f"\tERROR: A_chat_id value type is not an int in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True

        if type(mydict["B_chat_id"]) != int:
            print(f"\tERROR: B_chat_id value type is not an int in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True

        if type(mydict["A"]) != str: # or type(mydict["confidence"]) != float:
            print("#" * 50)
            print(f"\tERROR: A value type is not a string in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True

        if type(mydict["B"]) != str: # or type(mydict["confidence"]) != float:
            print("#" * 50)
            print(f"\tERROR: B value type is not a string in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True
        
        if type(mydict["ranking_criteria"]) != dict:
            print("#" * 50)
            print(f"\tERROR: the ranking_criteria is not a dict in dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True
        
        ####################################################################
        # Subdict keys
        subdict = mydict["ranking_criteria"]
        sub_remaining_set = required_ranking_keys - set(subdict.keys())
        if len(sub_remaining_set) > 0:
            missing_ranking_keys = ", ".join(sub_remaining_set)
            print("#" * 50)
            print(f"\tERROR: Missing the ranking_criteria keys [{missing_ranking_keys}] in the ranking_criteria sub-dictionary of dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True
            continue # stop checking this dict because it is missing fields
        
        ####################################################################
        # Subdict values
        # required_ranking_keys = set(["overall", "correctness", "relevance", "clarity", "completeness", "other"])
        for ranking_key in ["overall", "correctness", "relevance", "clarity", "completeness"]:

            if type(subdict[ranking_key]) != str:
                print("#" * 50)
                print(f"\tERROR: content value type is not a str in the ranking_criteria sub-dictionary of dictionary #{dict_cnt} in the json file {filepath}.")
                has_errors = True

            val = subdict[ranking_key]
            if val not in possible_ranking_values:
                possible_values_str = ", ".join(possible_ranking_values)
                print("#" * 50)
                print(f"\tERROR: value `{val}` isn't any of the following allowed roles [{possible_values_str}] in the ranking_criteria sub-dictionary of dictionary #{dict_cnt} in the json file {filepath}.")
                has_errors = True

        if type(subdict["other"]) != str:
            print("#" * 50)
            print(f"\tERROR: content value type is not a str in the ranking_criteria sub-dictionary of dictionary #{dict_cnt} in the json file {filepath}.")
            has_errors = True

        dict_cnt += 1

if not has_errors:
    print("Successfully passed format verification for demonstration json files.")

