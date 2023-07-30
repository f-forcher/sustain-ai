from pypdf import PdfReader
import datetime
import openai
import os
import sys
import tiktoken
import pandas as pd
import json
import glob
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
#from text_splitter_mod import TokenTextSplitter

openai.api_key = "***REMOVED***"


# def extract_all_text():
#     reader = PdfReader("./samples/bristol_2021-22.pdf")
#     str_list = []
#     for page in reader.pages:
#         str_list.append(page.extract_text())
#     return ''.join(str_list)


def read_pdf(filepath):
    """Takes a filepath to a PDF and returns a string of the PDF's contents"""
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\n\n Page N: {page_number} \n\n"
    return pdf_text

def get_themes(pdf_file_path, model_name = "gpt-3.5-turbo-16k-0613"):
    report_text = read_pdf(pdf_file_path)
    # enc = tiktoken.encoding_for_model(model_name)

    text_splitter = TokenTextSplitter(model_name=model_name,
    chunk_size=4000, chunk_overlap=400)
    texts = text_splitter.split_text(report_text)

    answers_analyze = []
    answers_json = []
    inputs = []
    #for chunk_enc in create_chunks_with_overlap(text, 4000, enc, overlap_ratio=0.1):
    #   chunk_text = enc.decode(chunk_enc)
    for chunk_text in texts:
        inputs.append(chunk_text)
        role = """
You are an incredibly advanced AI for text and sentiment analysis, 
able to understand all the themes of a text even when expressed more implicitly. Your answers 
may be used by an automated system, so *when asked* to format an answer to JSON using the token
"JSON_answer:" , it is very important that you only output the syntactically correct 
format and no other unformatted text. Otherwise, 
you can write out an extensive structured analysis to help explain your reasoning"""

        prompt_analyze = f"""
You are an advanced expert AI specializing in thematic and sentiment analysis. 
Your task is to identify presence or absence of McCowan's Five Modalities
in sustainability reports of some universities.

In the following quoted text, McCowan's Five Modalities of the University in Sustainable Development (Education, Knowledge production, Services, Public debate, Campus Operations) are explained.

```

Education covers the activities of teaching and learning in formal courses at undergraduate and graduate levels, 
as well as non-formal education in other spaces of the university (as explored in McCowan, 2021). 
Knowledge production covers research and scholarship of a ‘blue skies’ nature, as well as applied research, innovation 
and development of technology. There are broad range of activities encompassed by the modality ‘services’,1 designating 
those activities which directly serve citizens or support the work of other organisations or communities, for example 
provision of hospitals and legal clinics, professional development programmes, consultancy and secondments to government 
or the private sector. Public debate involves promoting spaces for deliberation, as well as the communication of research 
findings and political mobilisation. Finally, campus operations refer to the organisation of the physical university space, 
its staff and students, and the impacts that they have directly on the ecosphere.

Some more details: 
1) Education (personal development and personal civic learning). It considers inclusion of sustainability into curriculum, 
personal development and education for civic engagement. Quote from McCowan: 
"education refers to the role of the university as a space for learning, and for personal, civic and professional development."

2) Knowledge Production (basic research, technological innovation). 
Knowledge production implies generation of knowledge, and normally arises from research and scholarship 
carried out by academic staff, but in some instances also by students and community members. 
This modality includes not only basic and blue skies research, but also knowledge applied to the 
practical demands of government, industry and civil society organisations, the development of new 
forms of technology, and innovation more broadly.

3) Public Debate (dissimilation of ideas, deliberative space) 
It is a broader set of public engagement activities that relate to debates in the public sphere, 
through the ideas put forward in formal research outputs such as journal articles, 
which filter their way through the media into public discussion, 
or through the direct engagements of staff in the media or social media. 
In some cases, universities will have their own media outlets such as newsletters, blogs, radio, and even television stations.
This modality can also express itself through the political involvement of staff and students, 
their participation in campaigns and protests, and in other forms of direct action. 
Universities can also serve as sites (either physical or virtual) for hosting and encouraging 
deliberation and debate, as discussed by Marginson (2011) in relation to the ‘public sphere’ mode of the public good.

4) Service Delivery (outreach activities, secondments). 
There are services delivered directly to communities, for example running a health or legal clinic 
that community members can access, monitoring levels of air pollution to provide information when 
it is unsafe to go out, or running a short course on business French. 
This category also includes services provided to government, organisations and business, such as consultancy and secondments.

5) Campus Operations (sustainability planning, investments). 
As a community and as an organisation, the university manages its finances, its human resources, 
purchases equipment, uses fuel, sells food and merchandise, in some cases makes investments, 
and all of these activities have implications in terms of mitigation and adaptation of climate change. 
In some cases, universities own land beyond their immediate campuses, 
and make decisions about the usage of that land, for agriculture, forestry or commercial developments. 
In this category we would also include the travel undertaken by students and staff, 
a significant source of carbon emissions: while this might appear to be within the ‘education’ category, 
it is not strictly a result of the teaching and learning itself, but of the logistical organisation of the 
institution and its members. For some institutions the goal in terms of campus operations is to become carbon neutral 
or net zero, which can involve not only reducing emissions, but also offsetting through carbon credits or sequestering carbon. 
Campus operations can include net zero strategies, investments, waste management, canteen sourcing, sustainable procurement. 

```

Task 1:
You should read and analyse carefully the following chunk of text extracted from a sustainability report.
Then tell which of the five modalities or themes are present in the text. 
Write out an answer thoroughly and in detail, in a structured form. 

Task 2:
Another task is to analyze the chunk's sentiment. For our purposes would be sufficient to understand if there is any
self evaluation at all, and if there is any negative result reported such as missed target. 
for example missing previously set targets and similar stuff. Find negative or self evaluative statements. 
Again write out an answer thorougly and in a structured way, provide direct quotes if necessary.

Examples: 
- Self evaluation, negative: “We did not achieve this target, as our water consumption (m3) reduced by 4.3% from our 2005/06 baseline 
to our target year 2020/21”. 

- Self evaluation, negative: “Unfortunately we did not meet our water reduction target this year however we did achieve a reduction of 1.3% 
and are looking at ways of improving over the next year”. 

- Self evaluation, negative: “In 2019–20 we recycled 56% of our waste, meaning there remains a challenge to reach the 70% target.” 

- Self evaluation, neutral: “This year 56.4% of our waste was recycled. The recycling rate for operational and construction waste decreased this year. 
The re-tender of the University’s main waste collection contract has been completed which will increase recycling rates” . 

After the analysis for the two tasks, create a single-row table with eight column: a 
column for each of the five modality and a true/false if it is present or not, and two columns named 
"criticism" and "self-evaluation" with a true/false value if respectively criticism and "self-evaluation" are present or not. 
Last column should be a "validity" check: it may happen that the chunk is too small or does not contain any 
meaningful information (for example it may be a references page). If you really consider this to be the case, and
do not believe you can reliably extract any information related to the themes and sentiment, 
set the validity value and all the others to false.

Table template:
```
| Validity | Education | Knowledge Prod. | Services | Public Debate | Campus Ops | Self-evaluation | Criticism       | 
|----------|-----------|-----------------|----------|---------------|------------|-----------------|-----------------|
| BOOL     | BOOL      | BOOL            | BOOL     | BOOL          | BOOL       | BOOL            | BOOL            |
```

The chunk to analyze:
```
{chunk_text}
```

Your analysis:
"""

        promp_format = """
Now summarize these results and the table in JSON format, following this template:
```
{
    "valid": BOOL
    "analysis": {
        "education": BOOL, 
        "knowledge_production": BOOL, 
        "services": BOOL, 
        "public_debate": BOOL, 
        "campus_operations": BOOL,
        "self_evaluation": BOOL,
        "criticism": BOOL
    }
}
```

Example outputs:
```
{
    "valid": true
    "analysis": {
        "education": false, 
        "knowledge_production": true, 
        "services": true, 
        "public_debate": false, 
        "campus_operations": true,
        "self_evaluation": true,
        "criticism": true
    }
}

{
    "valid": false
    "analysis": {
        "education": false, 
        "knowledge_production": false, 
        "services": false, 
        "public_debate": false, 
        "campus_operations": false,
        "self_evaluation": false,
        "criticism": false
    }
}
```

You _must_ output JUST the json and nothing else.

JSON_answer:
        """
        # print(chunk_text, sys.stderr)
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt_analyze}
            ],
            temperature=0.,
        )
        print(response, sys.stderr)
        response_analyze_text = response['choices'][0]['message']['content']
        answers_analyze.append(response_analyze_text)
        
        # print(chunk_text, sys.stderr)
        response_json = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt_analyze},
                {"role": "assistant", "content": response_analyze_text},
                {"role": "user", "content": promp_format},
            ],
            temperature=0.,
        )
        print(response_json, sys.stderr)
        response_json_text = response_json['choices'][0]['message']['content']
        answers_json.append(response_json_text)
        
    return (answers_analyze, answers_json, inputs)

def flatten_json(json_response, uni, year, chunk, version=0, timestamp="NULL", model_name="NULL"):
    dict_response = json.loads(json_response)
    # Create a new dictionary with the 'uni', 'year', and 'chunk' keys
    flat_dict = {
        "uni":     uni,
        "year":    year,
        "chunk":   chunk,
        "version": version,
        "timestamp": timestamp,
        "model_name": model_name,
    }

    # Add the 'valid' key and value to the new dictionary
    flat_dict["valid"] = dict_response["valid"]

    # Add the keys and values from the 'analysis' dictionary to the new dictionary
    for key, value in dict_response["analysis"].items():
        flat_dict[key] = value

    return flat_dict

def analyze_one_report(uni, year, model_name="gpt-3.5-turbo-16k-0613"):
    pdf_file_path = f"./pdf_download/{uni}/{year}.pdf"
    if not os.path.exists(pdf_file_path):
        sys.stderr.write(f"The file {pdf_file_path} does not exists.\n")
        return
    (answers_analyze, answers_json, inputs) = get_themes(pdf_file_path=pdf_file_path, model_name=model_name)
    print(answers_json)
    
    dir_name = f"./outputs/analyses/{model_name}/{uni}/{year}"
    try:
        os.makedirs(dir_name, exist_ok=True)
        # print(f"Directory {dir_name} created.", sys.stderr)
    except OSError as error:
        sys.stderr.write(f"Error creating directory {dir_name}: {error}\n")
    
    #timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    filename = f"{dir_name}/out-{timestamp}.txt"
    
    # version is the number of out files already existing in the target folder
    version = len(glob.glob(f"{dir_name}/*.txt"))
    with open(filename, "a") as text_file:
        print(f"\n[INFO] Model name: {model_name}\n", file=text_file)
        print(f"\n[INFO] Pdf file name: {pdf_file_path}\n", file=text_file)
        print(f"\n[INFO] timestamp: {timestamp}\n", file=text_file)
        for i, js in enumerate(answers_json):
            print(f"\n[INFO] Chunk {i}:\n", file=text_file)
            print(f"\n[INFO] Chunk {i}, text:\n", file=text_file)
            print(inputs[i], file=text_file)
            print(f"\n[INFO] Chunk {i}, model answer:\n", file=text_file)
            print(answers_analyze[i], file=text_file)
            print(f"\n[INFO] Chunk {i}, json output:\n", file=text_file)
            print(js, file=text_file)
    
    answers_dict_flat = [flatten_json(answer.strip(), uni, year, chunk, version, timestamp, model_name) for chunk, answer in enumerate(answers_json)]
    
    return answers_dict_flat

def run():
    # report_link = "https://www.bristol.ac.uk/media-library/sites/green/UoB_SustainabilityReport_2122_FINAL.pdf"
    # model_name = "gpt-4"
    

    links_normalized = pd.read_csv("./reports_norm.csv", index_col=["HEI_names_norm"])
    
    print(links_normalized)
    
    unis = links_normalized.index.to_list()
    years = links_normalized.columns.to_list()
    model_name = "gpt-3.5-turbo-16k-0613"
    
    print(unis)
    print(years)
    
    for uni in unis:
        for year in years:
            sys.stderr.write(f"Analyzing report for {uni} year {year}\n")
            if pd.isna(links_normalized.loc[uni,year]):
                sys.stderr.write(f"Report for {uni} year {year} not present, skipping\n")
                continue
            
            answers_dict_flat = analyze_one_report(uni, year, model_name=model_name)
            if answers_dict_flat is None:
                continue
            print(answers_dict_flat)
            new_data = pd.DataFrame.from_records(answers_dict_flat).set_index(["uni" ,"year", "chunk", "version", "timestamp", "model_name"])
            
            old_data = None
            if os.path.exists("results.csv"):
                old_data = pd.read_csv("results.csv", index_col=["uni" ,"year", "chunk", "version", "timestamp", "model_name"])
            if old_data is not None:
                data = pd.concat([old_data,new_data], verify_integrity=True)
            else:
                data = new_data
            data.to_csv("results.csv")

if __name__ == "__main__":
    run()
