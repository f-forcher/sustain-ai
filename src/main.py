from pypdf import PdfReader
import openai

openai.api_key = "***REMOVED***"

def extract_all_text():
   reader = PdfReader("./samples/bristol_2021-22.pdf")
   str_list = []
   for page in reader.pages:
      str_list.append(page.extract_text())
   return ''.join(str_list)

def get_themes(text, open):
   report_text=extract_all_text() 
   prompt=f"""
Role: You are an incredibly advanced AI for text and sentiment analysis, able to understand all the themes of a text even when expressed more implicitly.

Question: In the next paragraph, McCowan's Five Modalities of the University in Sustainable Development (Education, Knowledge production, Services, Public debate, Campus Operations) are explained.

Education covers the activities of teaching and learning in formal courses at undergraduate and graduate levels, as well as non-formal education in other spaces of the university (as explored in McCowan, 2021). Knowledge production covers research and scholarship of a ‘blue skies’ nature, as well as applied research, innovation and development of technology. There are broad range of activities encompassed by the modality ‘services’,1 designating those activities which directly serve citizens or support the work of other organisations or communities, for example provision of hospitals and legal clinics, professional development programmes, consultancy and secondments to government or the private sector. Public debate involves promoting spaces for deliberation, as well as the communication of research findings and political mobilisation. Finally, campus operations refer to the organisation of the physical university space, its staff and students, and the impacts that they have directly on the ecosphere.

Can you read and analyse carefully the sustainability report (text follows) and tell me which of these five modalities or themes are present? Answer thoroughly and in detail, in a structured form. 

Report:
{report_text}

Your analysis:
"""

   response = openai.Completion.create(
      engine="gpt-3.5-turbo-16k-0613",
      prompt=prompt,
      temperature=0.5,
      max_tokens=1,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["\n\n\n"]
   )
   sentiment = response.choices[0].text.strip()
   return sentiment

def run():
   #report_link = "https://www.bristol.ac.uk/media-library/sites/green/UoB_SustainabilityReport_2122_FINAL.pdf"
   pdf_text = extract_all_text()
   themes_detailed = 
   pass

if __name__ == "__main__":
   run()
