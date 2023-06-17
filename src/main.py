from pypdf import PdfReader


def extract_all_text():
    reader = PdfReader("./samples/bristol_2021-22.pdf")
    for page in reader.pages:
        print(page.extract_text())
        print()
   

def run():
   OPENAI_API_KEY = "***REMOVED***"

   pass

if __name__ == "__main__":
   extract_all_text()