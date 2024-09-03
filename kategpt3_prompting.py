from utils import get_sciper
from gpt_wrapper.chat import Chat

from transformers import RobertaTokenizer, RobertaModel
import numpy as np
import torch
import tqdm

import json
import os
from copy import deepcopy
from typing import Literal


def mean_pooling(model_output, attention_mask):
    # https://github.com/jiachangliu/KATEGPT3/blob/9b8b8c77ecfee09a99a17ab0e03842749d173ee8/retrieval/kNN_preprocessing.py#L108
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_similarity(examples_embeddings:torch.Tensor, question_embedding:torch.Tensor, similarity_metric:Literal["euclidean", "cosine"]="euclidean")->torch.Tensor:
    """Calculate the simmilarity between a certain question and numerous examples using their sentence embeddings.

    Args:
        examples_embeddings (torch.Tensor): Tensor of shape (num_examples, embedding_dim)
        question_embedding (torch.Tensor): Tensor of shape (embedding_dim,)
        similarity_metric (Literal[&quot;euclidean&quot;, &quot;cosine&quot;], optional): Which simmilarity metric to use of the embeddings. Defaults to "euclidean".

    Returns:
        torch.Tensor: A simmiliarity score vector of shape (num_examples,)
    """
    if similarity_metric == "euclidean":
        similarity = -torch.cdist(examples_embeddings, question_embedding.unsqueeze(dim=0), p=2.0).squeeze()
    elif similarity_metric == "cosine":
        similarity = torch.nn.functional.cosine_similarity(examples_embeddings, question_embedding.unsqueeze(dim=0)).squeeze()
    return similarity

def main()->None:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large')
    
    examples_embeddings_already_calculated = os.path.exists(os.path.join("data", "examples_embeddings.pt"))
    sciper = get_sciper()
    
    with open(os.path.join("data", f"{sciper}.json"), "r") as f:
        questions = np.array(json.load(f))
    to_tokenize = [q["question_body"] for q in questions]
    
    if not examples_embeddings_already_calculated:
        with open(os.path.join("data", "examples.json"), "r") as f:
            examples = np.array(json.load(f))
        to_tokenize = [ex["question_body"] for ex in examples] + to_tokenize
        
    questions_tokenized = tokenizer(to_tokenize, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    with torch.no_grad():
        outputs = model(**questions_tokenized)

    questions_embeddings = mean_pooling(outputs, questions_tokenized['attention_mask'])

    if not examples_embeddings_already_calculated:
        nb_examples = len(examples)
        examples_embeddings, questions_embeddings = questions_embeddings[:nb_examples], questions_embeddings[nb_examples:]
        torch.save(examples_embeddings, os.path.join("data", "examples_embeddings.pt"))
    
    torch.save(questions_embeddings, os.path.join("data", f"{sciper}_questions_embeddings.pt"))


def find_example_types_indices(examples:list[dict])->tuple[np.ndarray, np.ndarray, np.ndarray]:
    open_examples = np.array([i for i, ex in enumerate(examples) if ex["question_options"] is None])
    tf_examples = np.array([i for i, ex in enumerate(examples) if ex["question_options"] is not None and len(ex["question_options"]) == 2])
    mcq_examples = np.array([i for i, ex in enumerate(examples) if ex["question_options"] is not None and len(ex["question_options"]) > 2])
    return open_examples, tf_examples, mcq_examples
    

def get_question_type(question:dict)->str:
    if question["question_options"] is None:
        question_type = "Open"
    elif len(question["question_options"]) == 2:
        question_type = "True/False"
    elif len(question["question_options"]) > 2:
        question_type = "Multiple Choice"
    return question_type

def gpt_generation(questions:list[dict], instructions:list[str]=["",""], examples_path:str = None, task_name:str="", model_args:dict={}, verbose:bool=False, subset_questions_indices:list[int]=None, nb_few_shot_examples:int=None)->list[dict]:
    if subset_questions_indices is None:
        subset_questions_indices = list(range(len(questions)))

    if examples_path is not None:
        with open(os.path.join(examples_path, f"examples.json"), "r") as f:
            examples = np.array(json.load(f))
        examples_embeddings = torch.load(os.path.join(examples_path, f"examples_embeddings.pt"))
        sciper = get_sciper()
        questions_embeddings = torch.load(os.path.join(examples_path, f"{sciper}_questions_embeddings.pt"))[subset_questions_indices]
        open_examples, tf_examples, mcq_examples = find_example_types_indices(examples)
        if nb_few_shot_examples is None:
            nb_few_shot_examples = len(examples)
        
    answers = []
    for qid, query in enumerate(questions[subset_questions_indices]):
        if verbose: print(f"Question {qid}".center(75, "-"))
        
        answers.append({key:query[key] for key in ['course_id', 'question_id']})
        answers[qid]["question"] = query["question_body"]
        answers[qid]["question_options"] = query["question_options"]
    
        question_type = get_question_type(query)
        
        few_shot_demonstration = ""
        
        if examples_path is not None:
            if question_type == "Open":
                relevant_examples = deepcopy(open_examples)
            elif question_type == "True/False":
                relevant_examples = deepcopy(tf_examples)
            elif question_type == "Multiple Choice":
                relevant_examples = deepcopy(mcq_examples)
                
            
            similarity = compute_similarity(examples_embeddings[relevant_examples], questions_embeddings[qid], similarity_metric="cosine")
            _, sorted_sim_indices = torch.sort(similarity, descending=True)
            examples_order = relevant_examples[sorted_sim_indices]
            
            for i, example in enumerate(examples[examples_order]):
                if i >= nb_few_shot_examples:
                    break
                few_shot_demonstration = "\n[SOLUTION] : " + example["answer"] + "\n" + few_shot_demonstration
                if question_type != "Open":
                    few_shot_demonstration = "\n[ ] " + "\n[ ] ".join(example["question_options"]) + few_shot_demonstration
                few_shot_demonstration = "[QUESTION] : " + example["question_body"] + few_shot_demonstration
                
            few_shot_demonstration += "[QUESTION] : "+ query["question_body"]
            if question_type != "Open":
                few_shot_demonstration += "\n[ ] " + "\n[ ] ".join(query["question_options"])
            few_shot_demonstration += "\n[SOLUTION] : "
        
        instructions_ = instructions.copy()
        
        zero_shot_demonstration = query["question_body"]
        if question_type != "Open":
            zero_shot_demonstration += "\n[ ] " + "\n[ ] ".join(query["question_options"])
        # if question_type == "True/False":
        #   instructions_[0] = instructions[0] + " If the answer is false, provide counter exemple, if the answer is true, provide a proof."
        #   instructions_[1] = instructions[1] + " If the answer is false, provide counter exemple, if the answer is true, provide a proof."
        # if question_type == "Multiple Choice":
        #     # instructions_[0] = instructions[0] + " At least one answer should be chosen."
        #     # instructions_[1] = instructions[1] + " At least one answer should be chosen."
        #     instructions_[0] = instructions[0] + " Explain each proposition one by one and justify your choice."
        #     instructions_[1] = instructions[1] + " Explain each proposition one by one and justify your choice."
        
        
        chatA = Chat.create(name=task_name+"_A_"+str(qid)) # create a new chat
        chatB = Chat.create(name=task_name+"_B_"+str(qid)) # create a new chat
        
        message_A = chatA.ask(few_shot_demonstration, instruction=instructions_[0], model_args=model_args)
        message_B = chatB.ask(zero_shot_demonstration, instruction=instructions_[1], model_args=model_args)
        
        answers[qid]["A_chat_id"] = message_A.chat_id
        answers[qid]["B_chat_id"] = message_B.chat_id
        answers[qid]["A"] = message_A.content.strip()
        answers[qid]["B"] = message_B.content.strip()
        
        if verbose:
            print("A".center(75, "-"))
            print(instructions_[0])
            print(few_shot_demonstration)
            print(answers[qid]["A"])
            print("B".center(75, "-"))
            print(instructions_[1])
            print(zero_shot_demonstration)
            print(answers[qid]["B"])
    return answers


if __name__ == "__main__":
    main()