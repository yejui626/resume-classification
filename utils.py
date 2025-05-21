from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.chat_models import ChatOpenAI
import uuid
import streamlit as st
import time
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import base64
from models import Candidate,Job
import pandas as pd
import os



def extract_information(file_path,job_title):
    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert extraction algorithm with 20 years experience in the recruiting industry. You will be provided with candidate's resume.
[Instruction] Extract relevant candidate's information mentioned in the following candidate's resume following the predefined properties.
1) Please provide accurate answers, no guessing.
2) Please return 'N/A' string for all the information that is not mentioned. Do not return NaN.
3) If current_location is missing, return 'Country': 'N/A', 'State': 'N/A', 'City': 'N/A' as a dict in a list. Do not return empty list.
5) For previous_job_roles , all of the keys (job_title,job_company,Industries,start_date,end_date,job_location,job_duration) must be present. Assign N/A to the values of the key if not mentioned.
4) Extracted Properties of all Start date and End date:
* if the month is not stated, assume that start/end date is in the middle of the year.
* should never include english words such as 'months', 'years', 'days'. 
* Instead, dates should be dates converted to the following format:
* date values assigned are strictly in Python datetime format.
Strict Format of either one: 
    YYYY
    YYYY-MM or YYYYMM
    YYYY-MM-DD or YYYYMMDD
4) Rules for job_duration calculation: 
* Any end date that indicates "Present", refers to {current_date}. 
* Method of duration calculation: Subtract the end date from start date to get the number of months. Finally sum up all relevant durations and convert to years. 
* Triple check your calculations.
* The result must be convertible to float value, encapsulate the value within double quotes, for example: "1.5".
"""
            ),
            ("human", 
             """[Job Title] 
{job_title}
[Candidate's Resume]
{text}   
"""),
        ]
    )

    loader = PyMuPDFLoader(file_path, extract_images=True)
    # loader = PyPDFLoader(file_path)
    documents = loader.load()

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    runnable = prompt | llm.with_structured_output(schema=Candidate)
    result = runnable.invoke({"job_title":job_title,"text": documents,"current_date":datetime.now()})

    return result


def define_criteria(save_dir,job_title,job_description,job_requirement,applicant_category):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert recruiting algorithm with 20 years experience in the recruiting industry. You will be provided with the job details (job title, applicant category, job description, job requirement). Execute all the tasks from step 1 to step 3 by strictly following the rules.
[Tasks]
1. Fill in relevant criteria's information based on the following job details with their properties.
2. If the criteria are not specified, you should apply your hiring knowledge to suggest details to the criteria.
3. Assign weightage to each of the criteria based on how important you feel they are in the job details.
[Rules]
- Make sure every criteria has one suggested detail.
"- Suggest at least one detail based on common market hiring criteria.
- You will penalized if you return 'Not Specified' as answer
- targeted_employer should be strictly returned in the format 'include(), exclude()', both include and exclude should be returned even if it's not mentioned.
"""
            ),
            ("human", 
            """[Job Details]
Job Title : {job_title}
Applicant Category : {applicant_category}
[Start of Job Description] 
{job_description} 
[End of Job Description]

[Start of Job Requirement]
{job_requirement}
[End of Job Requirement] """),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.1)
    runnable = prompt | llm.with_structured_output(schema=Job)
    result = runnable.invoke({"job_title":job_title,"job_description":job_description,"job_requirement":job_requirement,"applicant_category":applicant_category})

    criteria_data=[]
    weightage_data=[]
    
    for field_name in result.criteria[0].__fields__:
        criteria_data.append(getattr(result.criteria[0], field_name))
        
    for field_name in result.weightage[0].__fields__:
        weightage_data.append(getattr(result.weightage[0], field_name))

    df = pd.DataFrame({'details': criteria_data, 'weightage': weightage_data, 'selected':True})
    df.index = [x for x in result.criteria[0].__fields__]
    df.index.name='criteria'

    df.to_csv(os.path.join(save_dir, 'criteria.csv'))

    return df


def init_session(self, clear: bool =False):
    if not self.chat_inited or clear:
        st.session_state[self._session_key] = {}
        time.sleep(0.1)
        self.reset_history(self._chat_name)
