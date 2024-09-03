[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/zxIixFB4)
# CS-552 - Milestone 1

Welcome to the MNLP project! For M1, as you can read in the [project description](https://docs.google.com/document/d/1SP8SCHPOZZGEhs2ay-38FjedRE1bS9Q99VJb28eHoYk/edit?usp=sharing), you have 3 main goals: 
1. Collect preference data for 100 questions (individual work)
2. Review one paper (individual work)
3. Write the project proposal (group work)


## Repo Structure

The repo has 5 folders, 2 of which serve for you to submit all three deliverables:
1. `_artifacts` contains the [GPT Wrapper](https://docs.google.com/document/d/1MLi9dYJtPyN7QMyiutObozH1Y3naT_J_hs4J4OXw0cc/edit?usp=sharing) wheel so you can install it.
2. `_docs` contains the screenshots used below to explain how to trigger the tests manually.
3. `_templates` contains the latex templates for both your paper review as well as the project proposal. You MUST use these templates.
4. `_tests` contains some scripts which run automated tests so you can be sure your submission is correctly formatted.
5. `data` should be populated with your collected preference data. You should create a single file for all your 100 questions, name it `<YOUR-SCIPER>.json`. As such, when you submit there should be either 3 or 4 json files in this directory (depending on whether you're a group of 3 or 4).
6. `pdfs` should contain both your individual paper reviews (inside the `literature reviews` folder, with each one titled `<YOUR-SCIPER>.pdf`) and your project proposal (named `<YOUR-GROUP-NAME>.pdf`). This directoty should then have either 4 or 5 pdf files (where 1 is in the root `pdfs` directory, and 3/4 are in the `literature reviews` folder, depending on whether you're a group of 3 or 4).

## Running the tests manually
The autograding tests run automatically with every commit to the repo. If you want to trigger them manually, follow the instructions below:

1. Go to 'Actions' in the top bar
![go to 'Actions' in the top bar](_docs/1.png?raw=true)

2. Click on 'Autograding tests' on the left sidebar
![click on 'Autograding tests' on the left sidebar](_docs/2.png?raw=true)

3. Click on the 'Run workflow' button, next to 'This workflow has a workflow_dispatch event trigger.'
![click on the 'Run workflow' button, next to 'This workflow has a workflow_dispatch event trigger'](_docs/3.png?raw=true)

4. Confirm the test was succesfully passed
![confirm the test was succesfully passed](_docs/4.png?raw=true)

5. If it failed, check the details by clicking on that 'Autograding tests' entry, and then clicking on 'run-autograding-tests'. You can expand the sections to understand where the problem lies.
![you can expand the sections to understand where the problem lies](_docs/5.png?raw=true)

6. Note: if the JSON data validator fails with a `json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes` exception, your JSON has an error that prevents python from loading it in order to verify its format. Like the error indicates, check that the property names are enclosed in double quotes, but also that you don't have trailing commas (e.g., a comma after the last item of a dictionay), as they cause the same exception.
![JSON data validator failing with a json.decoder.JSONDecodeError](_docs/6.png?raw=true)