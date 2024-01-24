# sustain-ai
Repo for sustainability reports analysis.

Contribution for a social policy study exploring the differences between major UK universities' reports in various years using sentiment and thematic analysis, performed by ChatGPT-4.

# Summary
The sustainability reports are automatically retrieved from a collection of links, then `langchain` is used to split the text in chunks that are then sent to ChatGPT-4 API for analysis. A JSON is returned and parsed into a `Pandas` dataframe, which is then analysed.
