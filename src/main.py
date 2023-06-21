from pypdf import PdfReader
import datetime
import openai
import sys

openai.api_key = "***REMOVED***"

def extract_all_text():
    reader = PdfReader("./samples/bristol_2021-22.pdf")
    str_list = []
    for page in reader.pages:
        str_list.append(page.extract_text())
    return ''.join(str_list)

def get_themes(text):
    report_text=extract_all_text() 
    role = """You are an incredibly advanced AI for text and sentiment analysis, 
              able to understand all the themes of a text even when expressed more implicitly."""

    prompt=f"""
Question: In the next paragraph, McCowan's Five Modalities of the University in Sustainable Development (Education, Knowledge production, Services, Public debate, Campus Operations) are explained.

Education covers the activities of teaching and learning in formal courses at undergraduate and graduate levels, as well as non-formal education in other spaces of the university (as explored in McCowan, 2021). Knowledge production covers research and scholarship of a ‘blue skies’ nature, as well as applied research, innovation and development of technology. There are broad range of activities encompassed by the modality ‘services’,1 designating those activities which directly serve citizens or support the work of other organisations or communities, for example provision of hospitals and legal clinics, professional development programmes, consultancy and secondments to government or the private sector. Public debate involves promoting spaces for deliberation, as well as the communication of research findings and political mobilisation. Finally, campus operations refer to the organisation of the physical university space, its staff and students, and the impacts that they have directly on the ecosphere.

Can you read and analyse carefully the following sustainability report (it was extracted from
a pdf so be aware that it may be quite noisy)?
Then tell me which of the five modalities or themes are present in the report. 
Answer thoroughly and in detail, in a structured form. In the end create a table with a 
column for each modality and a yes/no if it is present or not.

Report:
{report_text}

Your analysis:
"""

    response = openai.ChatCompletion.create(
       model="gpt-3.5-turbo-16k-0613",
       messages=[
        {"role": "system", "content": role},
        {"role": "user", "content": prompt}
       ],
       temperature=0.,
    )
    print(response, sys.stderr)
    response_text = response['choices'][0]['message']['content']
    return response_text

def run():
    #report_link = "https://www.bristol.ac.uk/media-library/sites/green/UoB_SustainabilityReport_2122_FINAL.pdf"
    pdf_text = extract_all_text()
    response_text = get_themes(pdf_text)
    print(response_text)
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    filename = f"outputs/out-{timestamp}.txt"
    with open(filename, "w") as text_file:
        print(response_text, file=text_file)
    pass

if __name__ == "__main__":
    run()
