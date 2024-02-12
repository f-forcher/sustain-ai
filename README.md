# sustain-ai
Sentiment and thematic analysis of reports.

Contribution for a social policy study, performed using ChatGPT-4 API. Only the code is provided. not the input data files.

# Summary
Reports are automatically retrieved from a collection of links, then `langchain` is used to split the text in chunks that are then sent to ChatGPT-4 API for analysis. A JSON is returned and parsed into a `Pandas` dataframe, which is then analysed. An experimental attempt at RAG has been performed using `Weaviate` vector database.
