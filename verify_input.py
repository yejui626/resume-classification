import streamlit as st
import pandas as pd
import re
import os
from dotenv import load_dotenv

def validate_input_format(input_string): 
    pattern = r'^(include\((.*?)\)\s*,\s*exclude\((.*?)\)\s*)+$' # include special characters in company names eg &.
    
    if re.match(pattern, input_string):
        return True
    elif input_string == 'N/A':
        return True
    else:
        return False

def verify_input(verify_df):
    def is_not_specified(value):
        return value.lower() == 'not specified'
    
    try:
        # Validation for Academic Results (CGPA)
        temp = verify_df['details']['cgpa']
        if is_not_specified(temp):
            return True
        if temp == 'N/A':
            temp = 0
        float_value = float(temp)
    except:
        st.error("Error: Please make sure you only input digits for CGPA")
        return False

    try:
        # Validation for Skill Group
        values = [value.strip() for value in verify_df['details']['technical_skill'].split(',')]
        if values and not is_not_specified(values[0]):
            print(values)
    except:
        st.error("Error: Invalid input format for 'Skill Groups'. Input should follow the format: '[Value], [Value], [Value]'")
        return False

    try:
        # Validation for Years of Experience
        input_value = verify_df['details']['total_experience_year']
        if is_not_specified(input_value):
            return True
        if input_value == 'N/A':
            input_value = "0"

        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            st.error("Error: Invalid input format for 'Years of Total Work Experience'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
            return False
    except:
        st.error("Error: Invalid input format for 'Years of Total Work Experience'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
        return False
    
    try:
        # Validation for Years of Experience in Similar Role
        input_value = verify_df['details']['total_similar_experience_year']
        if is_not_specified(input_value):
            return True
        if input_value == 'N/A':
            input_value = "0"

        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            st.error("Error: Invalid input format for 'Years of Total Similar Work Experience'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
            return False
    except:
        st.error("Error: Invalid input format for 'Years of Total Similar Work Experience'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
        return False

    try:
        # Validation for Professional Certificate
        values = [value.strip() for value in verify_df['details']['professional_certificate'].split(',')]
        if values and not is_not_specified(values[0]):
            print(values)
    except:
        st.error("Error: Invalid input format for 'Professional Certificate'. Input should follow the format: '[Value], [Value], [Value]'")
        return False
    
    # Uncomment and modify if Candidate Current Location validation is needed
    try:
        # Validation for Candidate Current Location
        values = [value.strip() for value in verify_df['details']['candidate_current_location'].split(',')]
        if values and not is_not_specified(values[0]):
            print(values)
    except:
        st.error("Error: Invalid input format for 'Candidate Current Location'. Input should follow the format: 'Option 1: Country, Option 2 : State, Country, Option 3 : City, State, Country'")
        return False

    try:
        # Validation for Targeted Employer
        targeted_employer = verify_df['details']['targeted_employer']
        if is_not_specified(targeted_employer):
            return True
        if not validate_input_format(targeted_employer): 
            st.error("""Error: Invalid input format for 'Targeted Employer'. Input should follow the format: 'Example: True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments)", True - "include(), exclude()", True - "include(Shell, BP) , exclude()", True - "include() , exclude(Shell, BP)", False -  "include() , exclude(Shell," """)
            return False
    except:
        st.error("""Error: Invalid input format for 'Targeted Employer'. Input should follow the format: 'Example: True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments)", True - "include(), exclude()", True - "include(Shell, BP) , exclude()", True - "include() , exclude(Shell, BP)", False -  "include() , exclude(Shell," """)
        return False

    try:
        # Validation for Language
        values = [value.strip() for value in verify_df['details']['language'].split(',')]
        if values and not is_not_specified(values[0]):
            print(values)
    except:
        st.error("Error: Invalid input format for 'Language'. Input should follow the format: '[Value], [Value], [Value]'")
        return False

    try:
        # Validation for Expected Salary
        input_value = verify_df['details']['expected_salary']
        if is_not_specified(input_value):
            return True
        if input_value == 'N/A':
            input_value = "0"
        expected_format_patterns = [
            r'\d+$',  # Option 1: Number
            r'\d+-\d+$',  # Option 2: Number-Number
            r'>\d+$',  # Option 3: >Number
            r'<\d+$'  # Option 4: <Number
        ]
        if not any(re.match(pattern, input_value) for pattern in expected_format_patterns):
            st.error("Error: Invalid input format for 'Expected Salary in RM'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
            return False
    except:
        st.error("Error: Invalid input format for 'Expected Salary in RM'. Input should follow one of the formats: 'Option 1: Number', 'Option 2: Number-Number', 'Option 3: >Number', 'Option 4: <Number'")
        return False

    try:
        # Validation for Years of Graduation
        float_value = verify_df['details']['year_of_graduation']
        if is_not_specified(float_value):
            return True
        if float_value == 'N/A':
            float_value = "0"
        float_value = int(float_value)
    except:
        st.error("Error: Please make sure you only input the exact Year of Graduation. Example: '2024'")
        return False
    
    return True



