from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional,Dict
from datetime import datetime


# Pydantic Class for Candidate
class Candidate(BaseModel):
    """Information about a candidate from his/her resume."""

    name: Optional[str] = Field(..., description="The name of the candidate")
    phone_number: Optional[str] = Field(
        ..., description="The phone number of the candidate"
    )
    email: Optional[str] = Field(
        ..., description="The email of the candidate"
    )
    local: Optional[str] = Field(
        ..., description="Is the candidate Malaysian(Yes or No)? Identify it based on the country code of the phone number, address, and education background."
    )
    current_location: Optional[List] = Field(
        ..., 
        description="Candidate's current location/ address, assign the country based on the state and city. **You must return them in key-value pairs with Country, State, City as keys, even if they are not mentioned. Example: [{'Country': 'N/A', 'State': 'N/A', 'City': 'N/A'}] **",
        pattern = '\[\{\s*"Country"\s*:\s*"[^"]*"\s*,\s*"State"\s*:\s*"[^"]*"\s*,\s*"City"\s*:\s*"[^"]*"\s*\}\]'
    )
    education_background: Optional[List] = Field(
        ..., 
        description="Every single candidate's education background. (field_of_study, level (always expand to long forms), cgpa (Example: 3.5/4.0), university, start_date, year_of_graduation (Year in 4-digits only, remove month). Return in key-value pairs.",
        pattern = '\{\s*"field_of_study"\s*:\s*"[^"]*"\s*,\s*"level"\s*:\s*"[^"]*"\s*,\s*"cgpa"\s*:\s*"[^"]*"\s*,\s*"university"\s*:\s*"[^"]*"\s*,\s*"start_date"\s*:\s*"\d{4}"\s*,\s*"year_of_graduation"\s*:\s*"\d{4}"\s*\}'
    )
    expected_salary: Optional[str] = Field(
        ..., description="Candidate's expected salary in RM if known. (If the currency is Ringgit Malaysia, assign the numerical value or range values only Eg:'3000-3100'. If in other currency, assign alongside currency). Return 'N/A' if not found."
    )
    previous_job_roles: Optional[List] = Field(
        ..., description="Every single one of the candidate's (job_title, job_company, Industries (strictly classify according to to The International Labour Organization), start_date and end_date (only assign date time format if available. Do not assign duration), job_location, job_duration (return the job_duration in years and in string format), return in a python dict format. Assign N/A to the values of the key if not mentioned."
    )

    professional_certificate: Optional[List] = Field(
        ..., description="Candidate's professional certificates stated in the resume, return each certificate as a string in a python list. Return ['N/A'] if not found."
    )
    skill_group: Optional[List] = Field(
        ..., description="Every single candidate's skill groups, Technology (Tools, Program, System) stated in the resume, return each skills/ technology as a string in a python list. Return ['N/A'] if not found."
    )
    language: Optional[List] = Field(
        ..., description="Languages that is stated in the resume, return each language as a string in a python list. Return ['N/A'] if not found."
    )



# Pydantic Class for Criteria
class Criteria(BaseModel):
    """Hiring criteria based on the job details/ Suggested hiring criteria if not specified."""

    education_background: Optional[str] = Field(
        ..., description="Preferred education backgrounds. If not specified, suggest it based on the job title."
    )
    cgpa: Optional[str] = Field(
        ..., description="Minimum threshold of cgpa for the candidate required for the job. If not specified, suggest it."
    )
    technical_skill: Optional[str] = Field(
        ..., description="Technical skills that are relevant to the job details. If not specified, suggest it based on the job title. return the result in comma seperated form."
    )
    total_experience_year: Optional[str] = Field(
        ..., description="Minimum/preferred total years of working experience required for the job. If not specified, suggest it based on the applicant category."
    )
    professional_certificate: Optional[str] = Field(
        ..., description="Preferred professional certifications,licenses or accreditations required for the job. If not specified, suggest it based on the job title. return the result in comma seperated form."
    )
    total_similar_experience_year: Optional[str] = Field(
        ..., description="Minimum/preferred total years of working experience that is related to the job title required for the job. If not specified, suggest it based on the applicant category. Return the integer only."
    )
    language: Optional[str] = Field(
        ..., description="Preferred language required for the job. If not specified, suggest it. return the result in comma seperated form."
    )
    candidate_current_location: Optional[str] = Field(
        ..., description="Preferred candidate's current location required for the job. If not specified, suggest it."
    )
    targeted_employer: Optional[str] = Field(
        ..., description="Preferred inclusion or exclusion of candidate's previous company required for the job. **You must return them in this format: 'include(), exclude()' ** ",
        pattern = '^(include\((.*?)\)\s*,\s*exclude\((.*?)\)\s*)+$',
    )
    year_of_graduation: Optional[str] = Field(
        ..., description=f"Preferred year of graduation required for the job. If not specified, {datetime.now().year}"
    )
    expected_salary: Optional[str] = Field(
        ..., description="Preferred salary range required for the job. If not specified, search on glassdoor for the median salary range for the job role in Malaysia (RM)"
    )


# Pydantic Class for Criteria
class Weightage(BaseModel):
    """In the scale of 1-10, weightage assigned to the criteria based on the importance of the criteria in the job details"""

    education_background_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the education_background criteria"
    )
    cgpa_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the cgpa criteria"
    )
    technical_skill_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the technical_skill criteria"
    )
    total_experience_year_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the total_experience_year criteria"
    )
    professional_certificate_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the professional_certificate criteria"
    )
    total_similar_experience_year_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the total_similar_experience_year criteria"
    )
    language_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the language criteria"
    )
    candidate_current_location_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the candidate_current_location criteria"
    )
    soft_skill_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the soft_skill criteria"
    )
    year_of_graduation_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the year_of_graduation criteria"
    )
    expected_salary_weigh: Optional[str] = Field(
        ..., description="Weightage assigned to the expected_salary criteria"
    )


class Job(BaseModel):
    """Data about the job criteria and its weightage."""

    # Creates a model so that we can extract multiple entities.
    criteria: List[Criteria]
    # Creates a model so that we can extract multiple entities.
    weightage: List[Weightage]