import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper

# 1. Page Configuration
st.set_page_config(page_title="AI Market Intelligence Squad", page_icon="🕵️‍♂️", layout="wide")

# 2. Load environment variables
load_dotenv()

# --- FORCE INJECT API KEY ---
# This ensures sub-processes and CrewAI internal logic can see the key
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

api_key = os.environ.get("GROQ_API_KEY")

# 3. Define the Tool
@tool("wikipedia_search")
def wikipedia_search(search_query: str):
    """Search Wikipedia for historical facts, tech trends, and business data."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    return api_wrapper.run(search_query)

# 4. Streamlit UI Elements
st.title("🕵️‍♂️ Autonomous Market Intelligence Squad")
st.markdown("""
This system uses **Multi-Agent AI** to research, analyze, and report on any business or tech topic.
* **Agent 1:** Research (Wikipedia)
* **Agent 2:** Strategy Analyst
* **Agent 3:** Executive Editor
""")

topic = st.text_input("Enter a research topic:", placeholder="e.g., The evolution of NVIDIA GPUs in AI")

if st.button("🚀 Start Research"):
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found! Ensure your .env file has: GROQ_API_KEY=your_key_here")
    elif not topic:
        st.warning("Please enter a topic first.")
    else:
        with st.status("🤖 Agents are working...", expanded=True) as status:
            # A. Initialize LLM
            # We explicitly pass the key and set the model provider prefix
            my_llm = LLM(
                model="groq/llama-3.3-70b-versatile",
                api_key=api_key,
                temperature=0
            )

            # B. Define Agents
            st.write("👥 Gathering the squad...")
            researcher = Agent(
                role='Senior Market Researcher',
                goal=f'Uncover historical facts and data about {topic}.',
                backstory='Veteran researcher expert at extracting data from Wikipedia.',
                tools=[wikipedia_search],
                llm=my_llm,
                verbose=True
            )

            analyst = Agent(
                role='Lead Strategy Analyst',
                goal=f'Identify opportunities and risks based on research about {topic}.',
                backstory='Business strategist who turns facts into insights.',
                llm=my_llm,
                verbose=True
            )

            editor = Agent(
                role='Chief Technical Editor',
                goal='Format the analysis into a professional Markdown report.',
                backstory='Meticulous editor for executive-level reporting.',
                llm=my_llm,
                verbose=True
            )

            # C. Define Tasks
            research_task = Task(
                description=f'Find 3 major data points about {topic} using Wikipedia.',
                expected_output='A detailed summary of facts.',
                agent=researcher
            )

            analysis_task = Task(
                description='Analyze the research and write a strategic impact report.',
                expected_output='A strategic analysis report.',
                agent=analyst
            )

            editing_task = Task(
                description='Format the analysis into a clean Markdown report with headings.',
                expected_output='A polished final Markdown document.',
                agent=editor
            )

            # D. Execute Crew
            st.write("⚙️ Processing tasks (Research -> Analysis -> Editing)...")
            market_squad = Crew(
                agents=[researcher, analyst, editor],
                tasks=[research_task, analysis_task, editing_task],
                process=Process.sequential
            )
            
            result = market_squad.kickoff()
            status.update(label="✅ Research Complete!", state="complete", expanded=False)

        # E. Display Result
        st.subheader("🎯 Final Executive Report")
        st.markdown(result)
        
        # F. Download Option
        st.download_button(
            label="📥 Download Report as Markdown",
            data=str(result),
            file_name="market_report.md",
            mime="text/markdown"
        )