import os
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from transformers import pipeline

def run_research_agent(topic):
    print(f"Starting research on: {topic}\n")

    search_tool = DuckDuckGoSearchResults()
    wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki)

    search_results = search_tool.run(topic)
    wiki_results = wiki_tool.run(topic)

    combined_data = f"""
    Web Search Results:
    {search_results}

    Wikipedia Info:
    {wiki_results}
    """

    generator = pipeline("text-generation", model="google/flan-t5-base")

    prompt = f"""
    Write a clean and structured research report on: {topic}

    Use the data below:
    {combined_data}

    Make sure:
    - No repetition
    - Clear headings
    - Simple English

    Format strictly:

    Cover Page
    Title
    Introduction
    Key Findings
    Challenges
    Future Scope
    Conclusion
    """

    output = generator(prompt, max_new_tokens=300)

    result = output[0]["generated_text"]

    filename = topic.lower().replace(" ", "_") + "_report.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result)

    print("Report generated and saved!")

if __name__ == "__main__":
    run_research_agent("Impact of AI in Healthcare")