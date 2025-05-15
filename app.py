import streamlit as st
import os
import PyPDF2
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AlphaEvolve Paper Demonstration",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B8BBE;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #306998;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .highlight {
        background-color: #FFD43B;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Functions
@st.cache_data
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

@st.cache_data
def get_paper_analysis(pdf_text):
    """Analyze the paper content using LLM."""
    # Initialize the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    # Create a prompt template for academic summary
    summary_template = PromptTemplate(
        input_variables=["paper_text"],
        template="""
        You are an expert academic researcher with a PhD in computer science and machine learning.
        Analyze the following academic paper and provide:

        1. A comprehensive academic summary (800-1000 words)
        2. Key takeaways and contributions (5-7 points)
        3. Methodology overview
        4. Results and findings
        5. Limitations and issues identified in the paper
        6. Future research directions suggested

        Here is the paper text:
        {paper_text}

        Provide your analysis in a well-structured format with clear headings.
        """
    )

    # Create the chain
    summary_chain = LLMChain(llm=llm, prompt=summary_template)

    # Run the chain
    try:
        with st.spinner("Analyzing paper content..."):
            result = summary_chain.run(paper_text=pdf_text)
        return result
    except Exception as e:
        return f"Error analyzing paper: {str(e)}"

def get_agent_response(query, context):
    """Get response from the AI agent."""
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o")

    agent_template = PromptTemplate(
        input_variables=["query", "paper_context"],
        template="""
        You are a PhD-level AI research expert specializing in evolutionary algorithms,
        machine learning, and automated machine learning (AutoML). You have deep knowledge
        of the AlphaEvolve paper and its methodologies.

        Your task is to provide detailed, technically accurate responses to questions about
        the AlphaEvolve paper, explaining concepts at a PhD level while making them
        understandable. Use specific examples from the paper when relevant.

        Here is the context from the AlphaEvolve paper:
        {paper_context}

        User question: {query}

        Provide a comprehensive, technically precise answer that demonstrates deep expertise.
        Include relevant equations, algorithms, or technical details when appropriate.
        If you're unsure about any specific details, acknowledge the limitations of your knowledge.
        """
    )

    agent_chain = LLMChain(llm=llm, prompt=agent_template)

    try:
        with st.spinner("Generating response..."):
            result = agent_chain.run(query=query, paper_context=context)
        return result
    except Exception as e:
        return f"Error generating response: {str(e)}"

def run_experiment(experiment_type):
    """Simulate running an experiment based on AlphaEvolve methodology."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    if experiment_type == "basic":
        steps = [
            "Initializing population of program candidates...",
            "Setting up evolutionary operators...",
            "Configuring fitness function based on validation accuracy...",
            "Starting evolutionary search...",
            "Generation 1/10 complete...",
            "Generation 3/10 complete...",
            "Generation 5/10 complete...",
            "Generation 7/10 complete...",
            "Generation 10/10 complete...",
            "Evaluating best program on test set...",
            "Experiment complete!"
        ]
    else:  # advanced
        steps = [
            "Initializing population with diverse program architectures...",
            "Setting up multi-objective fitness evaluation...",
            "Configuring adaptive mutation rates...",
            "Implementing tournament selection with elitism...",
            "Starting evolutionary search with parallel evaluation...",
            "Generation 1/20 complete - Best fitness: 0.72...",
            "Generation 5/20 complete - Best fitness: 0.78...",
            "Generation 10/20 complete - Best fitness: 0.83...",
            "Generation 15/20 complete - Best fitness: 0.87...",
            "Generation 20/20 complete - Best fitness: 0.91...",
            "Performing ensemble selection from final population...",
            "Evaluating ensemble on test set...",
            "Experiment complete!"
        ]

    for i, step in enumerate(steps):
        # Update progress bar and status text
        progress = (i + 1) / len(steps)
        progress_bar.progress(progress)
        status_text.text(step)
        time.sleep(1)  # Simulate computation time

    # Generate results based on experiment type
    if experiment_type == "basic":
        results = {
            "Best validation accuracy": "0.85",
            "Test accuracy": "0.83",
            "Program complexity": "Medium",
            "Training time": "45 seconds",
            "Number of generations": "10",
            "Population size": "100"
        }
    else:  # advanced
        results = {
            "Best validation accuracy": "0.91",
            "Test accuracy": "0.89",
            "Program complexity": "High",
            "Training time": "120 seconds",
            "Number of generations": "20",
            "Population size": "200",
            "Ensemble size": "5",
            "Ensemble test accuracy": "0.92"
        }

    return results

# Main app
def main():
    # Sidebar
    st.sidebar.markdown('<div class="main-header">AlphaEvolve</div>', unsafe_allow_html=True)
    st.sidebar.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/images/Google_DeepMind_Logo.max-1000x1000.png", width=200)

    # Navigation
    page = st.sidebar.radio("Navigation", ["Paper Summary", "Key Takeaways", "Interactive Demo", "AI Agent Q&A"])

    # PDF path
    pdf_path = "D:\\OpenSourceLibrary\\AlphaEvolve\\AlphaEvolve.pdf"

    if os.path.exists(pdf_path):
        # Extract text from PDF (cached)
        pdf_text = extract_text_from_pdf(pdf_path)

        # Get paper analysis (cached)
        if "paper_analysis" not in st.session_state:
            st.session_state.paper_analysis = get_paper_analysis(pdf_text)

        analysis = st.session_state.paper_analysis

        # Display content based on selected page
        if page == "Paper Summary":
            st.markdown('<div class="main-header">AlphaEvolve: Academic Paper Summary</div>', unsafe_allow_html=True)

            # Display PDF viewer
            st.markdown('<div class="sub-header">Original Paper</div>', unsafe_allow_html=True)
            with st.expander("View Original PDF"):
                pdf_display = f'<iframe src="https://docs.google.com/viewer?url=https://storage.googleapis.com/deepmind-media/AlphaEvolve/AlphaEvolve.pdf&embedded=true" width="100%" height="600" frameborder="0"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            # Display summary
            st.markdown('<div class="sub-header">Comprehensive Summary</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section">{analysis}</div>', unsafe_allow_html=True)

        elif page == "Key Takeaways":
            st.markdown('<div class="main-header">Key Takeaways & Contributions</div>', unsafe_allow_html=True)

            # Extract key takeaways from the analysis
            if "Key takeaways" in analysis:
                takeaways_section = analysis.split("Key takeaways")[1].split("Methodology")[0]
                st.markdown(f'<div class="section">{"Key takeaways" + takeaways_section}</div>', unsafe_allow_html=True)

            # Extract limitations
            if "Limitations" in analysis:
                limitations_section = analysis.split("Limitations")[1].split("Future")[0] if "Future" in analysis else analysis.split("Limitations")[1]
                st.markdown('<div class="sub-header">Limitations & Issues</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section">{"Limitations" + limitations_section}</div>', unsafe_allow_html=True)

        elif page == "Interactive Demo":
            st.markdown('<div class="main-header">AlphaEvolve Interactive Demonstration</div>', unsafe_allow_html=True)

            st.markdown("""
            This interactive demonstration allows you to explore the core concepts of AlphaEvolve by running simulated experiments
            that follow the methodology described in the paper.
            """)

            demo_type = st.radio(
                "Select demonstration type:",
                ["Basic Evolutionary Search", "Advanced Multi-Objective Evolution"]
            )

            if st.button("Run Experiment"):
                experiment_type = "basic" if demo_type == "Basic Evolutionary Search" else "advanced"
                results = run_experiment(experiment_type)

                # Display results
                st.markdown('<div class="sub-header">Experiment Results</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                for i, (key, value) in enumerate(results.items()):
                    if i % 2 == 0:
                        col1.metric(key, value)
                    else:
                        col2.metric(key, value)

                # Visualization
                st.markdown('<div class="sub-header">Performance Visualization</div>', unsafe_allow_html=True)

                # Generate dummy data for visualization
                import numpy as np
                import pandas as pd
                import matplotlib.pyplot as plt

                generations = np.arange(1, 21 if experiment_type == "advanced" else 11)

                if experiment_type == "basic":
                    fitness = 0.5 + 0.35 * (1 - np.exp(-0.3 * generations))
                    data = pd.DataFrame({
                        "Generation": generations,
                        "Best Fitness": fitness + np.random.normal(0, 0.02, len(generations)),
                        "Average Fitness": fitness - 0.15 + np.random.normal(0, 0.03, len(generations))
                    })
                else:
                    fitness = 0.5 + 0.41 * (1 - np.exp(-0.2 * generations))
                    data = pd.DataFrame({
                        "Generation": generations,
                        "Best Fitness": fitness + np.random.normal(0, 0.02, len(generations)),
                        "Average Fitness": fitness - 0.2 + np.random.normal(0, 0.03, len(generations)),
                        "Best Ensemble": fitness + 0.05 + np.random.normal(0, 0.01, len(generations))
                    })

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data["Generation"], data["Best Fitness"], 'b-', label="Best Individual")
                ax.plot(data["Generation"], data["Average Fitness"], 'g-', label="Population Average")

                if experiment_type == "advanced":
                    ax.plot(data["Generation"], data["Best Ensemble"], 'r-', label="Ensemble Performance")

                ax.set_xlabel("Generation")
                ax.set_ylabel("Fitness (Validation Accuracy)")
                ax.set_title("Evolutionary Progress")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)

                # Code snippet
                st.markdown('<div class="sub-header">Sample Implementation</div>', unsafe_allow_html=True)

                if experiment_type == "basic":
                    code = """
                    # Basic AlphaEvolve-inspired implementation
                    def evolutionary_search(population_size=100, generations=10):
                        # Initialize population
                        population = [generate_random_program() for _ in range(population_size)]

                        for generation in range(generations):
                            # Evaluate fitness
                            fitness_scores = [evaluate_fitness(program) for program in population]

                            # Select parents
                            parents = tournament_selection(population, fitness_scores)

                            # Create new population
                            new_population = []
                            while len(new_population) < population_size:
                                # Apply genetic operators
                                if random.random() < 0.7:  # Crossover
                                    parent1, parent2 = random.sample(parents, 2)
                                    child = crossover(parent1, parent2)
                                else:  # Mutation
                                    parent = random.choice(parents)
                                    child = mutate(parent)

                                new_population.append(child)

                            population = new_population

                        # Return best program
                        fitness_scores = [evaluate_fitness(program) for program in population]
                        best_idx = np.argmax(fitness_scores)
                        return population[best_idx]
                    """
                else:
                    code = """
                    # Advanced AlphaEvolve-inspired implementation
                    def multi_objective_evolution(population_size=200, generations=20):
                        # Initialize diverse population
                        population = initialize_diverse_population(population_size)

                        # Track Pareto front
                        pareto_front = []

                        for generation in range(generations):
                            # Multi-objective evaluation (accuracy, complexity)
                            fitness_scores = [evaluate_multi_objective(program) for program in population]

                            # Update Pareto front
                            pareto_front = update_pareto_front(pareto_front, population, fitness_scores)

                            # Adaptive mutation rates based on population diversity
                            mutation_rate = calculate_adaptive_mutation_rate(population)

                            # Tournament selection with elitism
                            elite = select_elite(population, fitness_scores, elite_size=10)
                            parents = tournament_selection(population, fitness_scores)

                            # Create new population with elitism
                            new_population = elite.copy()

                            while len(new_population) < population_size:
                                # Apply genetic operators with adaptive rates
                                if random.random() < 0.65:  # Crossover
                                    parent1, parent2 = random.sample(parents, 2)
                                    child = crossover(parent1, parent2)
                                else:  # Mutation
                                    parent = random.choice(parents)
                                    child = mutate(parent, rate=mutation_rate)

                                new_population.append(child)

                            population = new_population

                        # Create ensemble from Pareto front
                        ensemble = create_ensemble(pareto_front, max_size=5)
                        return ensemble
                    """

                st.code(code, language="python")

        elif page == "AI Agent Q&A":
            st.markdown('<div class="main-header">Ask the AlphaEvolve Expert</div>', unsafe_allow_html=True)

            st.markdown("""
            Interact with our PhD-level AI agent that has deep knowledge of the AlphaEvolve paper.
            Ask technical questions about the methodology, results, or implications of the research.
            """)

            # User input
            user_question = st.text_input("Enter your question about AlphaEvolve:",
                                         placeholder="e.g., How does AlphaEvolve compare to other AutoML approaches?")

            if user_question:
                # Get response from agent
                response = get_agent_response(user_question, pdf_text)

                # Display response
                st.markdown('<div class="sub-header">Expert Response</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section">{response}</div>', unsafe_allow_html=True)

                # Suggest follow-up questions
                st.markdown('<div class="sub-header">Suggested Follow-up Questions</div>', unsafe_allow_html=True)

                # Generate follow-up questions based on the current question
                follow_ups = [
                    "How does AlphaEvolve handle the search space complexity?",
                    "What are the key innovations in AlphaEvolve compared to traditional genetic programming?",
                    "Can you explain the fitness evaluation process in AlphaEvolve?",
                    "What datasets were used to evaluate AlphaEvolve?",
                    "How does AlphaEvolve address the overfitting problem?"
                ]

                for question in follow_ups:
                    if st.button(question):
                        st.session_state.user_question = question
                        st.experimental_rerun()
    else:
        st.error(f"PDF file not found at {pdf_path}")

if __name__ == "__main__":
    main()
