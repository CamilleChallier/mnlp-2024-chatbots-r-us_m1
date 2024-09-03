from utils import *
import streamlit as st
import json
import os

# Read the inputs from the file with the 2 answers and all the other information
def read_inputs(file_path):
    with open(file_path, 'r') as file:
        inputs = json.load(file)
    return inputs

# Classify the answers based on several criteria
def classify_answers(question_data, question_index, total_questions):
    # Display the question and corresponding answers
    st.header(f"Question {question_index}/{total_questions}:")
    st.subheader("Question:")
    st.markdown(question_data["question"])

    if question_data["question_options"] is not None:
        st.subheader("Question options:")
        for option in question_data["question_options"]:
            st.write(option)

    st.subheader("Answer A:")
    st.markdown(question_data["A"])

    st.subheader("Answer B:")
    st.markdown(question_data["B"])
    
    # Initialize ranking criteria if not present
    if question_data.get("ranking_criteria") is None:
        question_data["ranking_criteria"] = {
            "overall": None,
            "correctness": None,
            "relevance": None,
            "clarity": None,
            "completeness": None,
            "other": {
                "Conciseness" : None,
                "Engagement": None
            }
        }
        
    # Classify the answers
    label_options = ["A", "B", "AB"]
    label_index_dict = {option:i for i, option in enumerate(label_options)}
    for criterion, label in question_data["ranking_criteria"].items():
        if criterion != "other":
            question_data["ranking_criteria"][criterion] = st.radio(
                f"{criterion.upper()}:", options=label_options, key=f"{question_index}_{criterion}",
                index = label_index_dict.get(label)
            )
        else:
            for sub_criterion, sub_label in question_data["ranking_criteria"]["other"].items():
                question_data["ranking_criteria"]["other"][sub_criterion] = st.radio(
                    f"{sub_criterion.upper()}:", options=label_options, key=f"{question_index}_{sub_criterion.lower()}",
                    index = label_index_dict.get(sub_label)
                )

    return question_data

# Main function
def main():
    # Get the SCIPER
    sciper = get_sciper()
    save_path = os.path.join("temp_data", f"{sciper}_annotated_data.json")

    st.title("Preference data annotation")

    # Load existing annotations or copy data from generated_questions.json if no existing annotations
    if "data" not in st.session_state:
        if os.path.exists(save_path):
            st.write("Load existing annotations...")
            with open(save_path, 'r') as file:
                data = json.load(file)
        else:
            st.write("No existing annotations found. Copying data from generated_questions.json...")
            data = read_inputs(os.path.join("temp_data", f"{sciper}_generated_answers.json"))
        st.session_state["data"] = data

    # Initialize question index
    question_index = st.session_state.get("question_index", 0)
    total_questions = len(st.session_state["data"])

    # Jump to a specific question
    jump_to_question = st.text_input("Jump to question number:", key="jump_to_question")
    jump_button_clicked = st.button("Jump to question")
    if jump_button_clicked and jump_to_question.isdigit():
        question_index = int(jump_to_question) - 1
        if question_index < 0:
            question_index = 0
        elif question_index >= total_questions:
            question_index = total_questions - 1
        st.session_state["question_index"] = question_index
    
    # If there are still questions to be answered
    if question_index < total_questions:
        # Classify the answers
        st.session_state["data"][question_index] = classify_answers(st.session_state["data"][question_index], question_index + 1, total_questions)

        # Save the annotated data
        if st.button("Save annotated data"):
            with open(save_path, 'w') as file:
                json.dump(st.session_state["data"], file, indent=4)
            st.success(f"Data saved to {save_path}")

        # Move to the next question
        if st.button("Next question") and question_index < total_questions-1:
            question_index += 1
            st.session_state["question_index"] = question_index
            st.experimental_rerun()
            
        # Previous question
        if st.button("Previous question") and question_index > 0:
            question_index -= 1
            st.session_state["question_index"] = question_index
            st.experimental_rerun()
            
        st.header(f"Question {question_index+1}/{total_questions}:")

if __name__ == "__main__":
    main()
