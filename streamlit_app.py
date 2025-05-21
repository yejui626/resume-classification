import streamlit as st
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
import os
from dotenv import load_dotenv
from utils import extract_information,define_criteria
from pdf_converter import doc2pdf
import secrets,string
import pandas as pd
from assess_criteria_class import JobParser, ResumeParser
import json
from default_chat import DefaultChat
from pandas_chat import PandasChat
from langchain.agents import AgentType, Tool, initialize_agent
from streamlit_pills import pills
from verify_input import verify_input
import re

load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"xxxx"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)


def format_info_field(extracted_info_dict, field_name):
    try:
        formatted_entries = []
        for i, entry in enumerate(extracted_info_dict.get(field_name, {}), start=1):
            output_string = f"{i}. "
            for key, value in entry.items():
                output_string += f"{key}: {value} | "  # Append key-value pairs
            formatted_entries.append(output_string)
        return '\n'.join(formatted_entries)
    except AttributeError:
        return ''
    
# Define the function
def format_info_field_evaluation(entries):
    try:
        formatted_entries = []
        if not entries or not isinstance(entries, list) or not entries:
            return entries

        for i, entry in enumerate(entries, start=1):
            output_string = f"{i}. "
            for key, value in entry.items():
                output_string += f"{key}: {value} | "  # Append key-value pairs
            formatted_entries.append(output_string)  # Remove trailing ' | '
        return '\n'.join(formatted_entries)
    except AttributeError:
        return entries
    
def replace_degrees(text):
    return re.sub(r"\b(Bachelor's|Master's)\b", lambda m: m.group(0).replace("'", ""), text)

def replace_s_but_not_start_date(text):
    # This regex looks for 's' that is not preceded by 'start_date'
    pattern = r"'s "
    return re.sub(pattern, " ", text)


def evaluate_criteria_pipeline(data_dict, criteria_df, resume_parser):
    total_score = 0
    data_dict['education_background'] = data_dict['education_background'].apply(replace_degrees)
    data_dict['education_background'] = data_dict['education_background'].apply(lambda x: json.loads(x.replace("'", '"')))
    data_dict['previous_job_roles'] = data_dict['previous_job_roles'].apply(replace_s_but_not_start_date)
    data_dict['previous_job_roles'] = data_dict['previous_job_roles'].apply(lambda x: json.loads(x.replace("'", '"')))
    data_dict['current_location'] = data_dict['current_location'].apply(lambda x: json.loads(x.replace("'", '"')))
    data_dict['language'] = data_dict['language'].apply(lambda x: json.loads(x.replace("'", '"')))
    data_dict['professional_certificate'] = data_dict['professional_certificate'].apply(lambda x: json.loads(x.replace("'", '"')))
    data_dict['skill_group'] = data_dict['skill_group'].apply(lambda x: json.loads(x.replace("'", '"')))

    for index, row in criteria_df.iterrows():
        details = row['details']
        weightage = row['weightage']
        selected = row['selected']
        
        if selected:
            function_name = f"evaluate_{index}_score"
            function = getattr(resume_parser, function_name)
            
            if index in ["total_experience_year", "total_similar_experience_year", "year_of_graduation", "targeted_employer"]:
                # Functions that return two values
                data_dict[[f"{index}", f"{index}_score"]] = data_dict.apply(lambda row: pd.Series(function(row, details, weightage)), axis=1)
                total_score += data_dict[f"{index}_score"]
            else:
                # Functions that return one value
                data_dict[f"{index}_score"] = data_dict.apply(lambda row: function(row, details, weightage), axis=1)
                total_score += data_dict[f"{index}_score"]

    # Add the gpt_recommendation_summary step
    data_dict['gpt_recommendation_summary'] = data_dict.apply(lambda row: resume_parser.gpt_recommendation_summary(row), axis=1)
    data_dict['total_score'] = total_score

    # Sort data_dict by total_score in descending order
    data_dict = data_dict.sort_values(by='total_score', ascending=False).reset_index(drop=True)
    
    return data_dict

# Define your Streamlit app
def main():
    st.set_page_config(page_title="Resume Parser", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Generative AI Resume Parser 💬🦙")
    st.info("Fill in this form before you start!", icon="📃")

    if "batch_token" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.batch_token = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

    with st.form("my_form",clear_on_submit=False,border=True):
        # st.title('Resume Parser Form')
        job_title = st.text_input(label=":rainbow[Job Title]", value="", on_change=None, placeholder="Insert your job title here", label_visibility="visible")
        job_description = st.text_area(label=":rainbow[Job Description]", value="", on_change=None, placeholder="Insert your job description here", label_visibility="visible")
        job_requirement = st.text_area(label=":rainbow[Job Requirement]", value="", on_change=None, placeholder="Insert your job requirement here", label_visibility="visible")
        applicant_category = st.selectbox(
            'Which applicant category does this job belong to?',
            ('Entry-Level', 'Mid-Level', 'Managerial-Level'))

        # Create a directory if it doesn't exist
        save_dir = f"uploaded_files/{st.session_state.batch_token}"
        os.makedirs(save_dir, exist_ok=True)

        # File uploader
        uploaded_files = st.file_uploader("Drop your resume here", accept_multiple_files=True, type=['docx','doc','pdf'])

        # Check if files are uploaded
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                file_name = uploaded_file.name
                file_path = os.path.join(save_dir, file_name)
                
                # Write the contents of the file to a new file
                with open(file_path, "wb") as f:
                    f.write(bytes_data)
                
                # Provide feedback to the user
                st.write("File saved:", file_name)
        else:
            print("FAILED")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")

    predefined_prompt_selected = pills("Q&A", ["What is the purpose of the Resume Parser tool?", 
                                               "What are the expected outputs for the extracted results?", 
                                               "What is the format for the evaluating criteria details that users should follow?"], 
                                               ["🍀", "🎈", "🌈"],clearable=True,index = None)

    if predefined_prompt_selected:
        prompt = predefined_prompt_selected
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                response = st.session_state.chat_engine.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response}) # Add response to message history

        
    if submitted:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    st.session_state["button_pressed"] = True
                    result_df = pd.DataFrame()
                    result_df_showcase = pd.DataFrame()
                    df = define_criteria(save_dir,job_title,job_description,job_requirement,applicant_category)

                    doc2pdf(save_dir)
                    # Get a list of PDF files in the directory
                    pdf_files = [filename for filename in os.listdir(save_dir) if filename.endswith(".pdf")]

                    for filename in pdf_files:
                        result = extract_information(os.path.join(save_dir, filename),job_title)
                        candidate_dict = result.dict()
                        st.write(candidate_dict)
                        # Convert the dictionary to a DataFrame
                        df = pd.DataFrame([candidate_dict])
                        df['file_name'] = filename
                        df_showcase_result = df.copy()

                        single_row_previous_jobs = format_info_field(candidate_dict, "previous_job_roles")
                        single_row_edu_background = format_info_field(candidate_dict, "education_background")
                        single_row_location = format_info_field(candidate_dict, "current_location")

                        df_showcase_result["previous_job_roles"] = single_row_previous_jobs
                        df_showcase_result["education_background"] = single_row_edu_background
                        df_showcase_result["current_location"] = single_row_location
                        
                        # Append the result DataFrame to the main DataFrame
                        result_df = pd.concat([result_df, df], ignore_index=True)
                        result_df_showcase = pd.concat([result_df_showcase, df_showcase_result], ignore_index=True)

                    # Display the Excel file in the chat and provide a download link
                    result_df.to_excel(os.path.join(save_dir, 'results.xlsx'))
                    st.session_state.messages = [{"role": "assistant", 
                                                "content": "Resume Parsing is done for all resumes! You may hover to the table below to download the results! If you wish to evaluate the candidates, please navigate to the sidebar!"}]
                    
                    message = {"role": "assistant", "content": result_df_showcase,"type":'dataframe'}
                    st.session_state.messages.append(message)
                except:
                    st.error("Please retry with valid resumes.")

    if "button_pressed" in st.session_state.keys():
        with st.sidebar:
            st.subheader('Define your evaluation criteria here!')
            st.markdown("You may edit the `details`, `weightage` and `selected` columns for this job evaluation.")
            st.markdown("Please refer to [this link](https://github.com/yejui626/fyp_goo?tab=readme-ov-file#3-what-is-the-format-for-the-evaluating-criteria-details-that-users-should-follow) for the criteria details format.")

            # Create DataFrame
            df = pd.read_csv(os.path.join(save_dir, 'criteria.csv'))

            edited_df = st.data_editor(df, disabled=['criteria'], key='df')

            verify_df = edited_df.set_index('criteria')
            # st.write(verify_df)
            if verify_input(verify_df):
                st.success("All inputs are valid!")

            favorite_command = edited_df.loc[edited_df["weightage"].idxmax()]["criteria"]
            selected_criterias = list(edited_df.loc[edited_df["selected"]]["criteria"])
            

            if st.button('Evaluate Now!'):
                 st.session_state["post_evaluation"] = False
                 with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        edited_df.to_csv(os.path.join(save_dir, 'criteria.csv'), index=False)

                        data_dict = pd.read_excel(os.path.join(save_dir, 'results.xlsx'),index_col=0)
                        criteria = pd.read_csv(os.path.join(save_dir, 'criteria.csv'),index_col=0)

                        # Initialize the JobParser class
                        job_parser = JobParser(job_title, job_description, job_requirement)

                        #Define embeddings model
                        embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002')

                        job_parser.extract_additional_skills()
                        job_parser.create_embeddings_for_jd_skills(embeddings_model, criteria['details']['technical_skill'])

                        # Initialize ResumeParser object with JobParser object
                        resume_parser = ResumeParser(job_title, job_description, job_requirement, job_parser)

                        # Run the pipeline
                        st.session_state.data_dict_final = evaluate_criteria_pipeline(data_dict, criteria, resume_parser)
                        st.session_state.data_dict_final.to_excel(os.path.join(save_dir, 'post_criteria_evaluation.xlsx'), index=False)

                        df_evaluation_showcase_result = st.session_state.data_dict_final.copy()

                        # Apply the format_info_field_evaluation function to each row in the specified columns
                        df_evaluation_showcase_result["previous_job_roles"] = df_evaluation_showcase_result["previous_job_roles"].apply(format_info_field_evaluation)
                        df_evaluation_showcase_result["education_background"] = df_evaluation_showcase_result["education_background"].apply(format_info_field_evaluation)
                        df_evaluation_showcase_result["current_location"] = df_evaluation_showcase_result["current_location"].apply(format_info_field_evaluation)

                        st.session_state["post_evaluation"] = True
                        st.session_state.messages.append({"role": "assistant", "content": "You can now start chatting with the results!","type":"message"})
                        st.session_state.messages.append({"role": "assistant", "content": df_evaluation_showcase_result ,"type":"message"})
                        
            st.markdown(f"Criteria : **{favorite_command}** has the highest weightage 🎈")
            st.markdown(f"Selected Criterias : **{selected_criterias}** 🎈")
                        
            st.divider()
                        
            cols = st.columns(1)
            if cols[0].button('Refresh'):
                print("session state before clearing",st.session_state)
                st.session_state.clear()
                print("session state after clearing",st.session_state)

    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        default_chat = DefaultChat()
        st.session_state.chat_engine = default_chat 

    if "post_evaluation" in st.session_state.keys():
        default_chat = DefaultChat()
        post_evaluation_chat = PandasChat(sav_dir=save_dir,batch_token=st.session_state.batch_token)
        tools = [
            #Structured
            Tool(
                name="Resume Evaluation Results DataFrame QA System",
                func=post_evaluation_chat.run,
                description="useful for when you need to answer questions about the information of the candidates or evaluation results dataframe. Input should be a fully formed question.",
            ),

            #Unstructured
            Tool(
                name="Resume Parser Tool QA System",
                func=default_chat.run,
                description="useful for when you need to answer questions about the resume parser tool itself. Input should be a fully formed question.",
            ),
        ]
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        st.session_state.chat_engine = agent 
        

    # Initialize the chat messages history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Fill in this form before you start!","type":"message"}]

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt,"type":"prompt"})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.run(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response}) # Add response to message history
                

if __name__ == "__main__":
    main()
