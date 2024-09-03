from utils import *
import streamlit as st
import json
import os


# read the inputs from the file with the 2 answers and all the other information
def read_inputs(file_path):
    with open(file_path, 'r') as file:
        inputs = json.load(file)
    return inputs


# classify the answers based on several criteria
def classify_answers(question_data, question_index, total_questions):
    st.header(f"Question {question_index}/{total_questions}:")

    # display the question and corresponding answers
    st.subheader("Question:")
    st.markdown(question_data["question"])

    if question_data["question_options"] is not None:
        st.subheader("Question options:")
        st.write("Question options:")
        for option in question_data["question_options"]:
            st.write(option)

    st.subheader("Answer A:")
    st.markdown(question_data["A"])

    st.subheader("Answer B:")
    st.markdown(question_data["B"])

    # create the dictionary with the classification criteria
    st.subheader("Classify the answers:")
    criteria = ['overall', 'correctness', 'relevance', 'clarity', 'completeness', 'conciseness', 'engagement']
    classifications = {criterion: None for criterion in criteria}

    for criterion in criteria:
        if criterion not in ['conciseness', 'engagement']:
            classification = st.radio(f"Classify based on {criterion}:", options=['A', 'B', 'AB'], key=f"{question_index}_{criterion}")
            classifications[criterion] = classification

    # classification of 'others' -> conciseness and engagement
    other_criteria = {}
    other_criteria['conciseness'] = st.radio(f"Classify based on 'conciseness':", options=['A', 'B', 'AB'], key=f"{question_index}_conciseness")
    other_criteria['engagement'] = st.radio(f"Classify based on 'engagement':", options=['A', 'B', 'AB'], key=f"{question_index}_engagement")

    other_classification = "; ".join([f"{criterion.capitalize()}: {classification}" for criterion, classification in other_criteria.items() if classification != 'None'])

    # save conciseness and engagement in 'other'
    classifications['other'] = other_classification

    return classifications


# main function
def main():

    # get the sciper
    sciper = get_sciper()

    st.title("Preference data annotation")

    #? put the input file with the correct SCIPER here
    questions = read_inputs(os.path.join("data", f"{sciper}_generated_answers.json"))
    question_index = st.session_state.get("question_index", 0)
    total_questions = len(questions)

    # add the field to jump to a certain question
    jump_to_question = st.text_input("Jump to question number:", key="jump_to_question")
    jump_button_clicked = st.button("Jump to question")
    # if the option to jump has been chosen and the inserted character is a number -> jump
    # otherwise -> nothing happens
    if jump_button_clicked and jump_to_question.isdigit():
        question_index = int(jump_to_question) - 1
        if question_index < 0:
            # start at the beginning
            question_index = 0
        elif question_index >= total_questions:
            # go to the last question
            question_index = total_questions - 1
        st.session_state["question_index"] = question_index

    # if there are still questions to be answered
    if question_index < total_questions:
        question_data = questions[question_index]

        # classify the answers
        classifications = classify_answers(question_data, question_index + 1, total_questions)

        # save the data
        if st.button("Save annotated data"):
            save_path = "annotated_data.json"

            # if there is already data that has been saved -> add the new data to the file
            if os.path.exists(save_path):
                with open(save_path, 'r') as file:
                    existing_data = json.load(file)
                existing_data.append({
                    "course_id": question_data["course_id"],
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "A_chat_id": question_data["A_chat_id"],
                    "B_chat_id": question_data["B_chat_id"],
                    "A": question_data["A"],
                    "B": question_data["B"],
                    "ranking_criteria": {key: value for key, value in classifications.items() if value is not None}
                })
                data = existing_data
            # if there is no data saved yet -> create a new file
            else:
                data = [{
                    "course_id": question_data["course_id"],
                    "question_id": question_data["question_id"],
                    "question": question_data["question"],
                    "A_chat_id": question_data["A_chat_id"],
                    "B_chat_id": question_data["B_chat_id"],
                    "A": question_data["A"],
                    "B": question_data["B"],
                    "ranking_criteria": {key: value for key, value in classifications.items() if value is not None}
                }]

            with open(save_path, 'w') as file:
                json.dump(data, file, indent=4)
            # if the data has been saved with success -> display a message
            st.success(f"Data saved to {save_path}")

        # next question button
        if st.button("Next question") and question_index < total_questions:
            question_index += 1
            st.session_state["question_index"] = question_index
            st.experimental_rerun()
        
        # previous question button
        if st.button("Previous question") and question_index > 0:
            question_index -= 1
            st.session_state["question_index"] = question_index
            st.experimental_rerun()


if __name__ == "__main__":
    main()


# run this file
# streamlit run data_annotation.py
