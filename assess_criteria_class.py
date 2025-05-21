from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import time
import ast
import os
from datetime import datetime
from openai import OpenAI
import numpy as np
import pandas as pd
import json
import openai
import math
import urllib3
from spacy.language import Language
from spacy_langdetect import LanguageDetector
import traceback

class JobParser:
    def __init__(self, job_title:str, job_description:str,job_requirement:str):
        self.job_title = job_title.lower()
        self.job_description = job_description
        self.job_requirement = job_requirement

    def extract_additional_skills(self):
        # Summarize job description
        client = OpenAI()
        response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125", #3.5 turbo
                messages=[
                    {"role": "system", "content": f"""Assume yourself as a hiring manager, you will be provided the job description and job requirement for {self.job_title}.
                     1. Extract the skills, technologies, programs, tools related to {self.job_title} from the text.
                     2. Output only the skills without any addiitonal reasonings.
                     3. The output result should strictly follows the python list format.
                     Example: ['Python','SQL','Hadoop']"""},
                    {"role": "user", "content": f"""[Start of Job Description]
                     {self.job_description}
                     [End of Job Description]
                     
                    [Start of Job Requirement]
                     {self.job_requirement}
                     [End of Job Requirement]
                     """}
                ],
            temperature=0,
            max_tokens=600,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0)
        jobdescription = response.choices[0].message.content
        
        jd_skills = ast.literal_eval(jobdescription)
        print("printing skills from jobdescription",jd_skills)
        
        self.jd_skills = jd_skills

        return self
    
    def create_embeddings_for_jd_skills(self, embeddings_model, result_list_skill):
        # Convert skills to lowercase
        jd_skills_lower = [x.lower() for x in self.jd_skills]
        result_list_skill_lower = [skill.strip().lower() for skill in result_list_skill.split(",")]

        # Create embeddings
        self.embedding_skill_groups = embeddings_model.embed_documents(jd_skills_lower + result_list_skill_lower)

        # Print the embeddings (or store them as needed)
        # print("Embeddings for JD skills and result list skills:", self.embedding_skill_groups)

        return self.embedding_skill_groups
    
    # def create_embeddings_for_technology(self, embeddings_model, result_list_tech):
    #     # Convert skills to lowercase
    #     result_list_tech_lower = [skill.strip().lower() for skill in result_list_tech.split(",")]

    #     # Create embeddings
    #     self.embedding_tech = embeddings_model.embed_documents(result_list_tech_lower)

    #     # Print the embeddings (or store them as needed)
    #     print("Embeddings for JD skills and result list skills:", self.embedding_tech)

    #     return self.embedding_tech
    

class ResumeParser:
    def __init__(self, job_title,job_description,job_requirement,job_parser):
        self.job_title = job_title
        self.job_description = job_description
        self.job_requirement = job_requirement
        self.job_parser = job_parser
        self.current_date = datetime.now()
        self.targEmp_industries_included = [] # from xlsx for 'included' ONLY

    # function to parse range inputs 
    def parse_range(self,input_string):
        """
        Parses the range string. 
        # # Example usage
        # input_string = "11.59-888"
        # lower_limit, upper_limit, condition = parse_range(input_string)

        Args:
        input_string: A string containing formats like "<5.6", ">5", "=5.0", or "2.0-5".

        Returns:
        tuple: A tuple containing the lower limit, upper limit, and condition.
        """
        match = re.match(r'^\s*(<|>|=)?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*-\s*([0-9]+(?:\.[0-9]+)?))?\s*$', input_string)
        condition = ""
        in_threshold_lower_limit = 0
        in_threshold_upper_limit = 99999

        if match:
            condition = match.group(1)
            values = match.group(2)

            if condition == "<":
                in_threshold_upper_limit = float(values)
            elif condition == ">":
                in_threshold_lower_limit = float(values)
            elif condition == "=":
                in_threshold_lower_limit = in_threshold_upper_limit = float(values)
            elif match.group(3): # range 
                condition = "range"
                in_threshold_lower_limit = float(values)
                in_threshold_upper_limit = float(match.group(3))
            else: # exact value, same as "="
                condition = "="
                in_threshold_lower_limit = in_threshold_upper_limit = float(values)
            # print(f"\tLower Limit: {in_threshold_lower_limit}, Upper Limit: {in_threshold_upper_limit}, Condition: {condition}")
            
        else:
            # print (f"\tVal = {input_string}  Parse Range funtion detected: Invalid input format")
            in_threshold_lower_limit, in_threshold_upper_limit = 0, 9999999

        return in_threshold_lower_limit, in_threshold_upper_limit, condition
    
    def run_evaluation(self, data_dict, criteria_df):
        for index, row in criteria_df.iterrows():
            if row['selected']:
                function_name = f"evaluate_{row['criteria']}_score"
                if hasattr(self, function_name):
                    eval_function = getattr(self, function_name)
                    eval_function(data_dict, row['details'], row['weightage'])

    def evaluate_education_background_score(self,data_dict, input, weightage):
        max_retries = 5
        retry_count = 0
        
        try:
            edu_prompt_system = f"""[Instruction] You will be provided with details such as the preferred field of study, job_title, and the candidate's field of study.
            Please act as an impartial judge and evaluate the candidate's field of study based on the job_title and preferred education background. For this evaluation, you should primarily consider the following accuracy:
            [Accuracy]
            Score 1: The candidate's field of study is completely unrelated to {input} and the job_title stated.
            Score 3: The candidate's field of study has minor relevance but does not align with {input} and the job_title stated.
            Score 5: The candidate's field of study has moderate relevance but contains inaccuracies to {input} and the job_title stated.
            Score 7: The candidate's field of study aligns with {input} and the job_title stated but has minor errors or omissions on either one of them.
            Score 10: The candidate's field of study is completely accurate and aligns very well with {input} and the job_title stated.
            
            [Rules]
            1. If the candidate has several education background, you should always consider the most related to {input} and the job_title only.
            2. You should always ignore those that are unrelated to {input} and the job_title and make sure they do not affect the total scoring.
            3. You should only assess the candidate's Field of Study and it's level. Ignore any other criterias.

            [Steps]
            Step 1 : You must rate the candidate on a scale of 1 to 10 by strictly following this format: "[[rating]]", 
            for example:
            "Education Background Rating: [[6]].

            [Question]
            How will you rate the candidate's education background based on the provided job_title with preferred education background?
            """

            edu_prompt_user = f"""
            Preferred Field of Study: {input}
            
            job_title: {self.job_title}

            [The Start of Candidate's Education Background]
            {data_dict['education_background']}
            [The End of Candidate's Education Background]
            """
            
            client = OpenAI()
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": edu_prompt_system},
                    {"role": "user", "content": edu_prompt_user}
                ],
                model="gpt-3.5-turbo-0125",
                temperature=0.3,
                n=3,
            )
            
            # print("Response from edu", response)
            
        except openai.RateLimitError as e:
            print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
            print(e)
            time.sleep(30)
            retry_count += 1

            if retry_count >= max_retries:
                print("Exceeded maximum retries for evaluating education background.... (From RateLimitError)")
                return response
        
        # Extract the number using regex
        def extract_gpt_response_rating(response):
            ratings = []
            pattern = r'\[\[([\d]+)\]\]'

            for i in range(len(response.choices)):
                match = re.search(pattern, response.choices[i].message.content)
                if match:
                    rating = int(match.group(1))
                    ratings.append(rating)
                else:
                    # ratings = 0
                    ratings.append(0)
            return ratings
        
        # Calculate average rating
        def calculate_average_rating(ratings):
            if not ratings:
                return 0
            return round(sum(ratings) / len(ratings))

        # Calculate weighted score
        def calculate_weighted_score(average_rating, weightage):
            if average_rating is None:
                return 0
            return round(average_rating / 10 * weightage)
                
        edu_rating = extract_gpt_response_rating(response)
        average_rating = calculate_average_rating(edu_rating)
        edu_weighted_score = calculate_weighted_score(average_rating, weightage)
        
        print(f"Candidate: {data_dict['name']}\t\t1. EDU Score:{edu_weighted_score}/{weightage}\t C: refer data_dict E: {input}\t ")
        
        return edu_weighted_score

    def evaluate_cgpa_score(self,data_dict,input_cgpa, weightage):
        out_weighted_cgpa_score = 0.0
        c_cgpa = 0 #total 

        def get_normalize_cgpa(cgpa_str,standard_scale = 4.0):
            # Regex pattern to match CGPA values and their max scales
            pattern = r'(\d+(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?'

            # Searching for the pattern in the text
            match = re.search(pattern, cgpa_str)
            if match:
                cgpa = float(match.group(1))
                max_cgpa = float(match.group(2)) if match.group(2) else standard_scale

                print(cgpa,max_cgpa)

                # Normalize CGPA to the standard scale
                normalized_cgpa = (cgpa / max_cgpa) * standard_scale
                print (f"""normalised cgpa:  {normalized_cgpa}, raw cgpa extracted: {cgpa_str}""")
                return normalized_cgpa
            else: # if N/A in resume, cpga -> 0.0 
                print ("normalised cgpa:  CPGA not found. Default CGPA = 0.0/4.0")
                return float("0")

        try:
            if 'education_background' not in data_dict:
                print(f"Candidate: {data_dict['name']}\t\t 2. CGPA Score:{out_weighted_cgpa_score}/{weightage}\t C CGPA(normalised): {c_cgpa} VS E: {input_cgpa} \t ")
                return 0.4 * weightage
            else: 
                print ("CGPA method 2: Getting latest available cgpa")
                data_list = ast.literal_eval(data_dict.education_background)
                # print(data_list)
                # print(data_list[0]['cgpa'])
                if data_list[0]['cgpa'] != 'N/A':
                    data_list.sort(key=lambda x: int(x['year_of_graduation']), reverse=True)
                    c_cgpa = get_normalize_cgpa(data_list[0]['cgpa'])

                if float(c_cgpa) >= float(input_cgpa):
                    out_weighted_cgpa_score = 1.0 * weightage
                else:
                    out_weighted_cgpa_score = 0.4 * weightage
                print(f"Candidate: {data_dict['name']}\t\t 2. CGPA Score:{out_weighted_cgpa_score}/{weightage}\t C CGPA(normalised): {c_cgpa} VS E: {input_cgpa} \t ")
        except Exception as e:
            print(e)
            out_weighted_cgpa_score = 0.4 * weightage
            print(f"Candidate: {data_dict['name']}\t\t 2. CGPA Score:{out_weighted_cgpa_score}/{weightage}\t C CGPA(normalised): {c_cgpa} VS E: {input_cgpa} \t ")

        return out_weighted_cgpa_score



    def evaluate_technical_skill_score(self,data_dict,input,weightage):
        data_dict_lower = [x.lower() for x in data_dict['skill_group']]
        print("skill groups:" , data_dict_lower)
                
        #Define embeddings model
        embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002')

        #Embeds both list
        embedding1 = embeddings_model.embed_documents(data_dict_lower) #candidate skill groups

        #Calculate the cosine similarity score from embeddings
        similarity_test = cosine_similarity(embedding1,self.job_parser.embedding_skill_groups)

        def similarity_range_score(similarity_scores):
            categorical_scores = []

            for score in similarity_scores:
                if score >= 0.88:
                    categorical_scores.append(1.0)
                elif score >= 0.85:
                    categorical_scores.append(0.5)
                elif score >= 0.8:
                    categorical_scores.append(0.3)
                else:
                    categorical_scores.append(0.0)
            print(categorical_scores)

            return categorical_scores

            
        res = round(np.mean(similarity_range_score(similarity_test.max(axis=0)))*weightage,2)
        
        print(f"Candidate: {data_dict['name']}\t\t3. SkillGroup Score:{res}/{weightage}\tC similairty score: {res} E: {input} \t ")
            
        return res

    def evaluate_total_experience_year_score(self,data_dict, input_string, weightage):
    
        c_total_yr_exp, out_weighted_score = 0.0, 0.0
        
        def calc_total_exp():
            # Check if 'previous_job_roles' exists and is a list
            
            total_duration = 0
            for role in data_dict['previous_job_roles']:
                try:
                    # Attempt to convert job duration to float and add to total
                    duration_str = role.get("job_duration", "0")  # Default to "0" if not found
                    duration = float(duration_str)
                    total_duration += duration
                except ValueError:
                    # Handle case where conversion to float fails
                    print(f"Error converting job duration to float for role: {role.get('job_title')}. Skipping this entry.")
                    continue  # Skip this entry and continue with the next
                    
            return round(total_duration, 2)

        # Manual: Total duration
        total_experience = calc_total_exp()
        # Use parse_range to get the lower and upper limits and condition
        in_threshold_lower_limit, in_threshold_upper_limit, condition = self.parse_range(input_string)
        try:
            c_total_yr_exp = float(total_experience)
            if c_total_yr_exp < in_threshold_lower_limit:
                out_weighted_score = 0  # does not meet requirement
            elif in_threshold_lower_limit <= c_total_yr_exp <= in_threshold_upper_limit:
                out_weighted_score = 1.0 * weightage  # within range ir equal 
            elif c_total_yr_exp > in_threshold_upper_limit:
                out_weighted_score = 0.5 * weightage  # overqualified
            else:
                out_weighted_score = 0
            print(f"Candidate: {data_dict['name']}\t\t4.Total years of experience Score:{out_weighted_score}/ {weightage}\t C:{c_total_yr_exp}, Required years: {input_string}\n ")
        except ValueError:
            # Handle the case where conversion to float fails
            out_weighted_score = 0  
        
        return total_experience,out_weighted_score


    def evaluate_total_similar_experience_year_score(self,data_dict, input, weightage):

        def extract_yoer_similar(data_dict):
            max_retries = 5
            retry_count = 0 
            try:
                yoer_prompt_system = f"""[Instruction] 
                You will be provided with details such as the candidate's previous job roles. Please act as a hiring manager with 20 years experience to evaluate the candidate's previous job roles.
                1. Identify job roles that are similar to {self.job_title}. You should also consider roles that are related to {self.job_title}.
                2. Return all of the duration of the related job roles into a python list.
                3. The output format should strictly follow the format in the example provided.
                Example of the output: Total duration: [[2,3,4]]

                [Question]
                What are the job durations for the job roles that are related to {self.job_title} in the candidate's previous job experience?
                """

                yoer_prompt_user = f"""
                Candidate's Previous Job Roles: {data_dict["previous_job_roles"]}
                """
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125", # 3.5 turbo
                    messages=[
                        {"role": "system", "content": yoer_prompt_system},
                        {"role": "user", "content": yoer_prompt_user}
                    ],
                    temperature=0.3,
                )
                # print("Response from yoer",response)
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
                print(e)
                time.sleep(30)
                retry_count += 1

                if retry_count >= max_retries:
                    print("Exceeded maximum retries for parsing PDF.... (From RateLimitError)")
                    return response


        def extract_duration(string):
            matches = re.findall(r'\[\[([0-9., ]+)\]\]', string)
            if matches:
                # Split by comma and directly convert each element to float
                list_of_floats = [float(x.strip()) for x in matches[0].split(",")]
                return list_of_floats
            else:
                print("No matches found for the pattern.")
                return []  # Fix to return a list directly

        def sum_floats_in_list(lst):
            if lst != 0:
                return math.fsum(lst)
            else:
                return 0

        def calculate_yoer(yoer_total, input_string, weightage):

            c_total_yr_exp = float(yoer_total)
            out_weighted_score = 0
            
            # Use parse_range to get the lower and upper limits and condition
            in_threshold_lower_limit, in_threshold_upper_limit, condition = self.parse_range(input_string)

            # Calculate the candidate's score based on their experience
            if c_total_yr_exp < in_threshold_lower_limit:
                out_weighted_score = 0  # does not meet requirement
            elif in_threshold_lower_limit <= c_total_yr_exp <= in_threshold_upper_limit:
                out_weighted_score = 1.0 * weightage  # within range ir equal 
            elif c_total_yr_exp > in_threshold_upper_limit:
                out_weighted_score = 0.5 * weightage  # overqualified
            else:
                out_weighted_score = 0


            return out_weighted_score


        response_yoer = extract_yoer_similar(data_dict)
        yoer_list = extract_duration(response_yoer)
        yoer_total = sum_floats_in_list(yoer_list)
        res = calculate_yoer(yoer_total, input, weightage)
        print(f"Candidate: {data_dict['name']}\t\t8. Yr of Exp in Role Score:{res}/{weightage}\t C: {yoer_total} E: {input}")
        
        return yoer_total,res

    def evaluate_current_location_score(self,data_dict, input, weightage):

        dataset_path = 'daerah-working-set.csv'
        city_data = pd.read_csv(dataset_path)

        def get_coordinates(city_name, country):
            # Try to get the coordinates from the dataset
            print("city name and country",city_name,country)
            try:
                city_info = city_data[city_data['Negeri'] == city_name]
                if country.lower() == "malaysia":
                    if city_info.empty==True:
                        city_info = city_data[city_data['Bandar'] == city_name]
                    latitude, longitude = city_info['Lat'].values[0], city_info['Lon'].values[0]
                    print("method1")
                    return latitude, longitude
            except IndexError:
                try:
                    http = urllib3.PoolManager(1, headers={'user-agent': 'cv_parser_geocoder'})
                    url = f'https://nominatim.openstreetmap.org/search?q={city_name}%2C+Malaysia&format=jsonv2&limit=1'
                    resp = http.request('GET', url)
                    loc = json.loads(resp.data.decode())
                    return loc[0]['lat'],loc[0]['lon']
                except:
                    return None,None
                

        def get_city_coast(latitude, longitude):
            east_coast_range =  (2.618, 6.2733, 101.3765, 103.6015)
            north_coast_range = (3.6857, 6.6999, 99.7166, 101.5265)
            middle_coast_range = (2.6884, 3.7801, 100.9878, 101.8911)
            south_coast_range =  (1.4645, 2.9702, 101.7863, 103.9107)
            east_malaysia_range = (1.0104, 6.9244, 109.7889, 119.0566)

            try:
                # Check which coast the city falls into
                if is_in_region(latitude, longitude, east_malaysia_range):
                    return "East Malaysia"
                elif is_in_region(latitude, longitude, middle_coast_range):
                    return "Middle Coast"
                elif is_in_region(latitude, longitude, east_coast_range):
                    return "East Coast"
                elif is_in_region(latitude, longitude, north_coast_range):
                    return "North Coast"
                elif is_in_region(latitude, longitude, south_coast_range):
                    return "South Coast"
                else:
                    return "Out of Malaysia"
            except TypeError:
                return "Location Not Detected"

        def is_in_region(latitude, longitude, region_range):
            min_lat, max_lat, min_lon, max_lon = region_range
            return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon
        
        state_mapping = {'wilayah persekutuan': 'WP', 'selangor': 'Selangor', 'johor': 'Johor', 'penang': 'Penang', 'pulau pinang': 'Penang', 'sabah': 'Sabah', 'sarawak': 'Sarawak', 'perak': 'Perak', 'kedah': 'Kedah', 'pahang': 'Pahang', 'terengganu': 'Terengganu', 'kelantan': 'Kelantan', 'negeri sembilan': 'N.Sembilan', 'melaka': 'Melaka','melacca': 'Melaka','perlis': 'Perlis'}
        
        def clean_state(data_dict):
            try:
                for key, value in state_mapping.items():
                    if key.lower() in data_dict['current_location']['State'].lower():
                        data_dict['current_location']['State'] = value
                        break
                return data_dict
            except:
                return data_dict

        def clean_location_string(location_str):
            try:
                # Split the string into city and country
                location_parts = list(map(str.strip, location_str.split(',')))

                # Handle the case when location_str only has city and country
                if len(location_parts) == 2:
                    state, country = location_parts

                    for key, value in state_mapping.items():
                        if key.lower() in state.lower():
                            state = value
                            break

                    city = 'N/A'
                elif len(location_parts) == 3:
                    city, state, country = location_parts

                    for key, value in state_mapping.items():
                        if key.lower() in state.lower():
                            state = value
                            break
                else:
                    country = location_parts[0]
                    state = 'N/A'
                    city = 'N/A'

                # Create the result dictionary
                result = {'Country': country, 'State': state, 'City': city}

                return result
            except ValueError:
                return location_str
        
        def evaluate_coordinate(cleaned_location,data_dict):
            #Get coordinates for required location and candidate location
            latitude1, longitude1 = get_coordinates(cleaned_location['State'],cleaned_location['Country'])
            print(latitude1, longitude1)
            latitude2, longitude2 = get_coordinates(data_dict['current_location']['State'], data_dict['current_location']['Country'])
            print(latitude2, longitude2)
            #Define the coast of required location and candidate location
            coast1 = get_city_coast(latitude1, longitude1)
            coast2 = get_city_coast(latitude2, longitude2)
            #Located at the same region(coast)
            if coast1 == coast2:
                return weightage*0.5
            #Located at different region
            else:
                return 0


        def evaluate_location(cleaned_location,data_dict,weightage):
            try:
                print(cleaned_location)
                print(data_dict['current_location'])
                # If candidate is in Malaysia
                if cleaned_location['Country'].lower() == "malaysia" and data_dict['current_location']['Country'].lower() == "malaysia":
                    # If Option 1 in excel
                    if cleaned_location['State'].lower() == 'n/a' and cleaned_location['City'].lower() == 'n/a':
                        return weightage
                    
                    # If same state
                    elif (data_dict['current_location']['State'].lower() == cleaned_location['State'].lower()):
                        # State = N/A
                        if cleaned_location['State'].lower() == 'n/a':
                            if cleaned_location['City'].lower() == 'n/a':
                                return 0
                            else:
                                print("weightage here")
                                return weightage
                        # State != N/A
                        else:
                            return weightage
                        
                    # if not same state
                    elif (data_dict['current_location']['State'].lower() != cleaned_location['State'].lower()):
                        # same city
                        if (data_dict['current_location']['City'].lower() == cleaned_location['City'].lower() == "N/A"):
                            return 0
                        else:
                            return evaluate_coordinate(cleaned_location,data_dict)
                        
                    # if same city
                    elif (data_dict['current_location']['City'].lower() == cleaned_location['City'].lower()):
                        # City = N/A
                        if cleaned_location['City'].lower() == 'n/a':
                            return 0
                        else:
                            print("weightage here")
                            return weightage
                    else:
                        return 0
                        
                # If candidate is overseas
                else:
                    if data_dict['current_location']['Country'] == cleaned_location['Country']:
                        print(cleaned_location['Country'],data_dict['current_location']['Country'])
                        return weightage
                    else:
                        return 0
            except TypeError as e:
                print("Different Country detected")
                print(e)
                return 0

        # Example usage:
        cleaned_location = clean_location_string(input)
        cleaned_dict = clean_state(data_dict)
        out_location_score =  evaluate_location(cleaned_location,cleaned_dict,weightage)
        print (f"Candidate: {data_dict['name']}\t\t 11. Location Score: {out_location_score}/{weightage}\t  E:{cleaned_location} C: {data_dict['current_location']}\n")
        return out_location_score



    def evaluate_targeted_employer_score (self,data_dict, in_target_employer, in_weightage_employer): 
        out_targetted_employer_score =  0

        # parse into include and excluded target comapanies 
        included_input = []
        excluded_input = []
        exclusion_match = ""

        def validate_input_format(input_string): 
            """
            Check if CVMATCHING template format correct for 
            Example: 
                True - "include(Shell, BP) ,  exclude( KLCC, Novella Clinical, Fidelity Investments)    
                True - "include(), exclude()"   
                True - "include(Shell, BP) , exclude()"    
                True - "include() , exclude(Shell, BP)" 
                False -  "include() , exclude(Shell, " 

            """
            # Regular expression pattern to match the valid format
            # pattern = r'^(include\([\w\s,]*\)\s*,\s*exclude\([\w\s,]*\)\s*)+$'
            pattern = r'^(include\((.*?)\)\s*,\s*exclude\((.*?)\)\s*)+$' # include special characters in company names eg &.
            
            # Check if the input string matches the pattern
            if re.match(pattern, input_string):
                return True
            else:
                return False
            
        def parse_targemp_input (correct_format_inputstring):    
            '''
                include_input updates to space-removed values 
                excluded_input updates to space-removed values 

                Example: 
                included_input: ['PetronasDigital']
                excluded_input: ['KLCC', 'NovellaClinical', 'FidelityInvestments']

                Reasoning: 
                More robust matching when spaces are removed. ExxonMobil matches Exxon Mobil inputted by User 
            ''' 
            # Regular expression pattern to match the include and exclude sections
            pattern = r'(include|exclude)\((.*?)\)'
            matches = re.findall(pattern, correct_format_inputstring)

            for match in matches:
                action, values = match
                values_list = [value.strip().replace(" ", "") for value in values.split(',')]
                
                if action == 'include':
                    included_input.extend(values_list)
                elif action == 'exclude':
                    excluded_input.extend(values_list)
            return True 

        # Preprocessing input & resume employers 
        def clean_employer_lst(input_str):
            """
            removes common words for better string matching 
            """
            # List of common words to remove
            common_words_to_remove = ["sdn", "bhd", "berhad", "ptd", "ltd", "inc", "co", "llc", "or", "and", "&"]
            pattern = r'\b(?:' + '|'.join(re.escape(word) for word in common_words_to_remove) + r')\b|-|\s+'
            cleaned_str = re.sub(pattern, '', input_str, flags=re.IGNORECASE)
            cleaned_list = [word.strip() for word in cleaned_str.split(',')]
            return cleaned_list
        
        def extract_indsutries (gpt_response): 
            """
            Extract industries from customised gpt output response. Example: 
                gpt_response = "[[Marketing, Food & Beverage, Shipping, Fashion, Cosmetics]]"
                output = ["Marketing", "Food & Beverage", "Shipping", "Fashion", "Cosmetics"]
            """
            # Ensure input is a string and follows the expected format
            if not isinstance(gpt_response, str) or not gpt_response.startswith("[[") or not gpt_response.endswith("]]"):
                return ["Unknown"]

            # Extract the content inside the outer brackets and split by comma
            # The slice [2:-2] removes the outermost brackets "[[" and "]]"
            industries = [industry.strip() for industry in gpt_response[2:-2].split(',')]

            return industries
        def get_employer_industries_gpt4 (company_name, company_location = ""): 
            max_retries = 5
            retry_count = 0
            # Classify employer industry by gpt
            system_p = f"""You are a helpful assistant. Given a company name and details, your task is to classify the given company's industry it is involve in as per The International Labour Organization.
            1. Classify the industries the company falls into according to The International Labour Organization, based on the company. 
            2. Output only all of industries in python list.
            3. The output format should strictly follow the format in the example provided below - enclosed with double brackets, comma-seperated
            4. A company can be classified in more than 1 industries. 
            Example of the output:
                example 1:  [[Marketing, Food & Beverage, Shipping, Fashion, Cosmetics]]
                example 2: [[Finance]]
                example 3: [[Unknown]] if the company is unfamiliar or you are unsure, output this. 

            """
            in_target_employer_petronas_description = "Petronas is a Malaysian oil and gas company that is involved in upstream and downstream activities. It is the largest oil and gas company in Malaysia, with operations in more than 30 countries around the world. Petronas is involved in exploration, production, refining, marketing, trading, and distribution of petroleum products. It also has interests in petrochemicals, shipping, engineering services, power generation, and other related businesses."
            p_example = f'[The Start of Company description] {in_target_employer_petronas_description}[The End of Company description] '
            p_example_response_format = "[[Oil and Gas, Petrochemicals, Refining, Retail, Shipping, Exploration and Production, Engineering and Construction]]"
            

            p = f'Classify the industries according to The International Labour Organization of the given company. Return results in the aforementioned output format. Given Company: {company_name}, located at {company_location}'
            try:
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125", 
                    messages=[
                        {"role": "system", "content": system_p},
                        {"role": "user", "content": p_example},
                        {"role": "assistant", "content": p_example_response_format},
                        {"role": "user", "content": p}
                    ]
                )
                try:
                    result = response.choices[0].message.content
                    industries_lst = extract_indsutries (result) if result else None 
                    print (f'GPT response on industry: {result}\tEXTRACTED INDUSTRIES: {industries_lst}')
                    return industries_lst
                except KeyError:
                    return "undectected"
            except openai.RateLimitError as e:
                print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
                print(e)
                time.sleep(30)
                retry_count += 1

                if retry_count >= max_retries:
                    print("Exceeded maximum retries for parsing PDF.... (From RateLimitError)")
                    return response
            except Exception as ire:
                print("InvalidReqError",ire)
                return "undetected"

        
        def check_if_matching_employer_industry():
            '''
                Input: 
                    user_input_bool: True if input is a list (ie from CVMatching xlsx since can be >1 company)
                    in_target_employer: Company Name
                Used when candidate has not work in target employer specified, check for matching industries: 
                    1. Ask GPT to classify the industries based on this description 
                    2. Check against candidate's previous job company industries
                    3. if candidate worked in similar industries: 50%, else 0%
            '''
            # variables
            out_targetted_employer_score =  0


            if (self.targEmp_industries_included == []): 
                init_input_employer_industry()

            candidate_industries = data_dict["Industries"]
            
            # find matches between overall industries and included()
            list1 = [x.lower().replace(" ", "") for x in candidate_industries if x.lower().replace(" ", "") != "unknown"]   
            list2 = [x.lower().replace(" ", "") for x in self.targEmp_industries_included if x.lower().replace(" ", "") != "unknown"]
            matches = [x for x in list1 if x in list2]

            print(f"GPT-ed Classified Industries.\t Included:{included_input} \tExcluded {excluded_input}. Included Industries{self.targEmp_industries_included}\t Candidate's data_dict['Industries']: {candidate_industries}\t Matched industries are: {matches}")
            if matches:
                print (f"Candidate: {data_dict['name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t Result: Case 2: Matching Industries are {matches}\n")
                res_employer = f"Matching industries detected: {matches}"
                res_employer_score = 0.5*float(in_weightage_employer)
                return res_employer,res_employer_score
            else:
                print (f"Candidate: {data_dict['name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t Result: Case 3: NO MATCHING INDUSTRY \n ")
                res_employer = f"No exact match and no matching industry from past employers detected"
                res_employer_score = 0
                return res_employer,res_employer_score

        def worked_in_excluded(candidate_employers, excluded): 
            excluded_matches = []
            for x in candidate_employers:
                    if x in excluded:
                        excluded_matches = f"Exclusion detected[{x}]"
                        return excluded_matches, True 
            return excluded_matches, False      
        
        def init_input_employer_industry ():
            """
            Initialises list for related-industries in criteria file by user
            
            """ 
            target_employer_industries = set()
            for employer in included_input:
                print(f"xlsx included employer {employer}")
                if employer: 
                    a = get_employer_industries_gpt4(employer)
                    target_employer_industries.update(a)
            # Assuming 'target_employer_industries_lst' is a list of lists (each inner list contains industries for an employer)
            self.targEmp_industries_included = list (target_employer_industries)
            print (f"RESUME PARSER CLASS INTIALISED: XLSX Target Employer related googlesearch industries.\tIncluded {included_input}\t self.targEmp_industries: {self.targEmp_industries_included}\n")
            return True 
                    
        # User/Employer Template input validation
        try:
            # Assuming validate_input_format raises an exception if validation fails
            if not validate_input_format(in_target_employer):
                raise ValueError("CVMatching Template.xlsx input string is invalid at 12.Target Employer")

            # If validation passes, proceed with parsing
            parse_targemp_input(in_target_employer)  # included, excluded is updated
            print(f"included: {included_input}, \t excluded: {excluded_input}")
        except ValueError as e:
            # Handle the validation error
            error_message = f"Warning, 12. Target Employer in file CVMatching Template.xlsx {e}"
            print(error_message)
            # self.targEmp_exclusion_matched = "CVMatching Template.xlsx input string is invalid at 12.Target Employer"
            return -1
            
        # Preprocessing inputs 
        req_employers = clean_employer_lst("".join(included_input))  # clean input from excel from common words 
        candidate_employers = clean_employer_lst(",".join([role["job_company"] for role in data_dict["previous_job_roles"] if isinstance(role, dict)]))
        print(f"12. Evaluating Target Employer\tIncluded: {included_input} \t excluded: {excluded_input}\tCandidate's previous employers: {candidate_employers}")

        # Preprocessing Data_dict of candidate: Reassign GPT classified industries for candidate's each previous employer 
        overall_industries = set()
        for x in data_dict["previous_job_roles"]:
            if isinstance(x, dict):
                q = x['job_company'] 
                l = x["job_location"] 
                industries_list = get_employer_industries_gpt4(q, l)  # This now returns a list directly
                
                # Directly assign the list without splitting
                x["Industries"] = industries_list
                
                # Update overall industries without needing to split; handle single-value lists correctly
                overall_industries.update([j.strip().strip('.') for j in industries_list])
                
                # Adjust the print statement to directly use industries_list
                print(f"{q} located at {l} is gpt-classified as a company in industries: {industries_list}")
                
        data_dict["Industries"] = list(overall_industries)
        # Scoring Method
        # 1. Check if candidate work in excluded companies 
        exclusion_match, excluded_flag = worked_in_excluded(candidate_employers, excluded_input)
        if excluded_flag:
            return exclusion_match,0
        else:
            # 2. Check for exact match with cleaned lists (employer and user)
            matched_employer = []
            for candidate in candidate_employers:
                # Skip if candidate is empty or whitespace
                if not candidate.strip():
                    continue

                for required in req_employers:
                    if re.search(fr'{re.escape(candidate)}', required, re.IGNORECASE):
                        matched_employer.append(candidate)
                        break # breaks right after matching 1 employer
                    
            if not matched_employer:# 3: Check for related industry in candidate's past employers 
                print ('\t...12. Target Employer: Checking for any past employers matching to industry of target employer')
                return check_if_matching_employer_industry()
                
            else: # exact match employer 
                print (f"Candidate: {data_dict['name']}\t\t 12. Targeted Employer Score: {out_targetted_employer_score}/{in_weightage_employer}\t  Result: Case 1: MATCHING EMPLOYER \t Matches = {matched_employer}\n)")
                res_employer =  f"Inclusion detected: {matched_employer}"
                res_employer_score = float(in_weightage_employer)
        return res_employer,res_employer_score

    def check_custom_languages(self,input_list, custom_languages):
        result = []
        for lang in input_list:
            normalized_lang = lang.strip().lower()
            for key, value in custom_languages.items():
                if key.lower() in normalized_lang:
                    normalized_lang = value
                    break
            result.append(normalized_lang.capitalize())
        return result

    def normalize_languages(self,input_list):
        # Define mappings for custom languages to a standardized form
        custom_languages_malay = {
            "bahasa melayu": "malay",
            "bahasa malaysia": "malay",
            "malay": "malay",
            "melayu": "malay",
            "bahasa": "malay",
        }

        custom_languages_mandarin = {
            "chinese": "mandarin",
            "huayu": "mandarin",
            "mandarin": "mandarin",
        }
        # Normalize Malay languages
        input_list = self.check_custom_languages(input_list, custom_languages_malay)
        # Normalize Mandarin languages
        input_list = self.check_custom_languages(input_list, custom_languages_mandarin)
        return input_list


    def evaluate_language_score(self,data_dict, input, weightage):
        print(data_dict['language'])
        match_percentage = 0
        try:
            candidate_list = data_dict['language']
            print(candidate_list)
            input_list = [lang.strip() for lang in input.split(",")]

            nlp = spacy.load('en_core_web_md')

            # Normalize custom languages for both Malay and Mandarin
            candidate_list = self.normalize_languages(list(candidate_list))
            input_list = self.normalize_languages(list(input_list))

            doc1 = nlp(str(candidate_list))
            doc2 = nlp(str(input_list))
            
            languages1 = set(ent.text.strip() for ent in doc1.ents if ent.label_ == "LANGUAGE")
            languages2 = set(ent.text.strip() for ent in doc2.ents if ent.label_ == "LANGUAGE")

            print("languages1", languages1)
            print("languages2", languages2)

            matched_languages = set(lang.lower() for lang in languages1).intersection(set(lang.lower() for lang in languages2))

            # Calculate the percentage of matches
            if languages2:
                match_percentage = len(matched_languages) / len(languages2) * 100
            else:
                match_percentage = 0
            
            language_score = round(match_percentage / 100 * weightage)
            print("Matched Languages: ", matched_languages)
            print(f"Candidate: {data_dict['name']}\t\t 14. Language Score: {language_score}/{weightage}\t C:{languages1} {candidate_list}, E: {input} \n")
            
            return language_score
            
        except Exception as e:
            print("Error on language", e)
            traceback.print_exc()  # This will print the traceback information
            return 0
        

    def evaluate_expected_salary_score(self,data_dict, input, weightage):
        """
        Checks if the candidate's expected salary matches the employer's range.

        Args:
        in_salary (str): Employer's expected salary range.
        c_exp_salary (str): Candidate's expected salary.

        Returns:
        int: Score indicating the match percentage.
        """
        # Assign 0 score for N/A or empty values
        if np.isnan(data_dict['expected_salary']):
            out_salary_score = 0
        else: 
            # Parse employer's expected salary range
            in_exp_sal_llimit, in_exp_sal_ulimit, in_exp_sal_condition = self.parse_range(input)

            # Parse candidate's expected salary, calculate average if it's a range
            c_exp_sal = 0 # default is 0 
            c_exp_sal_llimit, c_exp_sal_ulimit, c_exp_sal_condition = self.parse_range(data_dict['expected_salary'])
            if c_exp_sal_llimit != c_exp_sal_ulimit:
                # Alternative: Calculate average for a range
                    # c_exp_sal = (c_exp_sal_llimit + c_exp_sal_ulimit) / 2  
                c_exp_sal = c_exp_sal_llimit # assume lower limit when cv states sal range for now 
            else:
                c_exp_sal = c_exp_sal_llimit  # Use lower limit as single value if cv input not a range

            # Check if the candidate's expected salary falls within the employer's range
            if in_exp_sal_llimit <= c_exp_sal <= in_exp_sal_ulimit:
                res = 1  # 100% 
            else:
                res = 0
            
            out_salary_score = res * weightage

        print (f"Candidate: {data_dict['name']}\t\t 15. Exp Salary in RM Score: {out_salary_score}\t Employer: {input}, Candidate: {data_dict['expected_salary']}\n ")

        return out_salary_score

    def evaluate_professional_certificate_score(self,data_dict, input, weightage):
        def detect_match_phrases(resume, match_phrases):
            matches = []
            for certificate in resume:
                for phrase in match_phrases:
                    if isinstance(phrase, list):
                        for x in phrase:
                            pattern = re.compile(fr'\b{re.escape(x)}\b', re.IGNORECASE)
                            matches.extend(pattern.findall(certificate.lower()))
                    else:
                        pattern = re.compile(fr'\b{re.escape(phrase)}\b', re.IGNORECASE)
                        matches.extend(pattern.findall(certificate.lower()))

            # Remove duplicates by converting the list to a set and back to a list
            unique_matches = list(set(matches))

            return unique_matches
        
        def evaluate_candidate_score(matched_phrases, match_phrase_input, weightage):
            # Calculate the score based on the weightage
            score = round(len(matched_phrases) / len(match_phrase_input) * weightage,2)

            return score
        
        
        # Read the abbreviation CSV file
        # file_path = os.path.join(os.path.dirname(__file__), 'CVMatching_Prof_Cert_Wikipedia.xlsx')
        file_path = 'CVMatching_Prof_Cert_Wikipedia.xlsx'
        abb_csv = pd.read_excel(file_path)
        abb_csv = abb_csv[['Name', 'Abbreviation']]
        abb_csv = abb_csv.dropna(subset=['Abbreviation']).reset_index(drop=True)

        abb_csv['Name_lower'] = abb_csv['Name'].str.lower()
        unique_elements = [ue.strip() for ue in input.split(",")]

        # Retrieve 'Professional Certificate' field from data_dict
        professional_certificates = data_dict['professional_certificate']
        
        for phrase in unique_elements.copy():
            # Convert the current phrase to lowercase for case-insensitive comparison
            phrase_lower = phrase.lower()
            
            # Check if the lowercase phrase is an exact match in any lowercase entry in the 'Name' or 'Abbreviation' columns
            match = abb_csv[(abb_csv['Name_lower'] == phrase_lower) | (abb_csv['Abbreviation'].str.lower() == phrase_lower)]
            
            # If there is a match, remove both the abbreviation and the full name from the unique elements
            if not match.empty:
                # Update with matched abbreviations and names
                unique_elements.append([match['Name'].values[0],match['Abbreviation'].values[0]])
                unique_elements.remove(phrase)

        # Convert data_dict to a string
        data_dict_str = ''.join(professional_certificates)

        # Detect matched phrases
        matched_phrases = detect_match_phrases(data_dict_str, unique_elements)
        print('matched_phrases', matched_phrases,len(matched_phrases))
        print('unique_elements', unique_elements,len(unique_elements))
        # Evaluate candidate score
        score = evaluate_candidate_score(matched_phrases, unique_elements, weightage)
        
        print (f"Candidate: {data_dict['name']}\t\t 10. Prof Cert Score: {score}/{weightage}\t Employer's Certs: {input},  Candidate's Certs: {professional_certificates}\n ")
        
        return score

    def evaluate_year_of_graduation_score(self,data_dict, input_year, weightage): 
        out_yr_grad  =  0

        try:

            if 'education_background' not in data_dict:
                print("No educational background provided.")
                return "No educational background provided.",0
            else: 
                # Sort education background by year of graduation once, after preprocessing
                data_list = ast.literal_eval(data_dict.education_background)
                print(data_list)
                data_list.sort(key=lambda x: int(x['year_of_graduation']) if x['year_of_graduation'] != "N/A" else 0, reverse=True)
                # Preprocess and validate year of graduation entries
                res = ""
                for edu in data_list:
                    year_of_graduation = str(edu['year_of_graduation'])  # Ensure it's a string for comparison
                    print(year_of_graduation)
                    if not year_of_graduation.isdigit() and year_of_graduation.lower() not in ['present', 'current']:
                        edu['year_of_graduation'] = 'N/A'
                    elif year_of_graduation.lower() in ['present', 'current']:
                        edu['year_of_graduation'] = 'Still Studying'

                    year_of_graduation = str(edu['year_of_graduation']) 
                    res += year_of_graduation + ", "
                    if year_of_graduation == input_year:
                        out_yr_grad = weightage

                # Print the result
                res = res if res else "Not Specified" 
                print(f"16. Year of Grad: {out_yr_grad}\t Employer: {input_year},  Candidate: {res}")
        except Exception as e:
            print(e)
            res = "Not Specified"
            out_yr_grad = 0
            print(f"16. Year of Grad: {out_yr_grad}\t Employer: {input_year},  Candidate: {res}")

        return res,out_yr_grad
    

    def gpt_recommendation_summary(self,data_dict):
        max_retries = 5
        retry_count = 0
        
        data_df = pd.DataFrame.from_dict([data_dict])
        df = data_df[['education_background', 'skill_group',
        'previous_job_roles', 'professional_certificate', 'language']]
        candidate_info = df.to_dict()
        
        try:
            yoer_prompt_system = f"""[Instruction]
            You are the {self.job_title} recruiter, state all the alignments and misalignments of the candidate's qualifications and experience with the job description, and job requirements.

            [Question]
            - [Job Description]
            {self.job_description}
            - [Job Requirements]
            {self.job_requirement}
            """

            yoer_prompt_user = f"""
            {candidate_info}
            """
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125", # 3.5 turbo 
                messages=[
                    {"role": "system", "content": yoer_prompt_system},
                    {"role": "user", "content": yoer_prompt_user}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content
        except openai.RateLimitError as e:
            print(f"OpenAI rate limit exceeded. Pausing for one minute before resuming... (From RateLimitError)")
            print(e)
            time.sleep(30)
            retry_count += 1
            response = "Error"

            if retry_count >= max_retries:
                print("Exceeded maximum retries for parsing PDF.... (From RateLimitError)")
                return response



        