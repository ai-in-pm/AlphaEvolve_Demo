import streamlit as st
from alpha_evolve_demo import AlphaEvolveDemo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AlphaEvolve Demo",
    page_icon="🧬",
    layout="wide"
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
    .code-box {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Main app
st.markdown('<div class="main-header">AlphaEvolve Demonstration</div>', unsafe_allow_html=True)

st.markdown("""
This demonstration shows a simplified version of the AlphaEvolve methodology described in the paper.
It uses an LLM to generate, evaluate, and evolve code solutions to a given problem.
""")

# Problem selection
st.markdown('<div class="sub-header">Select Problem</div>', unsafe_allow_html=True)

problem_options = {
    "palindrome": """
    Create a function to find the longest palindromic substring in a given string.
    A palindrome is a string that reads the same backward as forward.

    Example:
    Input: "babad"
    Output: "bab" or "aba" (both are valid)

    Input: "cbbd"
    Output: "bb"
    """,

    "matrix_multiplication": """
    Create an efficient function to multiply two matrices A and B.
    The function should handle matrices of different sizes, checking if they are compatible for multiplication.

    Example:
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    Result = [[19, 22], [43, 50]]
    """,

    "sorting": """
    Create an efficient sorting algorithm that sorts an array of integers.
    Focus on optimizing both time and space complexity.

    Example:
    Input: [5, 2, 9, 1, 5, 6]
    Output: [1, 2, 5, 5, 6, 9]
    """
}

selected_problem = st.selectbox(
    "Choose a problem to solve:",
    options=list(problem_options.keys()),
    format_func=lambda x: x.replace("_", " ").title()
)

st.markdown('<div class="sub-header">Problem Description</div>', unsafe_allow_html=True)
st.markdown(f'<div class="code-box">{problem_options[selected_problem]}</div>', unsafe_allow_html=True)

# Evolution parameters
st.markdown('<div class="sub-header">Evolution Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    population_size = st.slider("Population Size", min_value=3, max_value=10, value=5, step=1)
with col2:
    generations = st.slider("Number of Generations", min_value=1, max_value=5, value=3, step=1)

# Run evolution
if st.button("Run AlphaEvolve"):
    with st.spinner("Running evolutionary process..."):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create a placeholder for logs
        log_output = st.empty()

        # Create AlphaEvolve instance
        alpha_evolve = AlphaEvolveDemo(
            problem_description=problem_options[selected_problem],
            population_size=population_size,
            generations=generations
        )

        # Override print function to capture logs
        import builtins
        original_print = builtins.print
        logs = []

        def custom_print(*args, **kwargs):
            output = " ".join(map(str, args))
            logs.append(output)
            log_output.markdown(f'<div class="code-box">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
            original_print(*args, **kwargs)

            # Update progress based on log content
            if "Initializing population" in output:
                progress_bar.progress(0.1)
                status_text.text("Initializing population...")
            elif "Evaluating initial population" in output:
                progress_bar.progress(0.2)
                status_text.text("Evaluating initial population...")
            elif "Generation 1/" in output:
                progress_bar.progress(0.4)
                status_text.text("Running generation 1...")
            elif "Generation 2/" in output:
                progress_bar.progress(0.6)
                status_text.text("Running generation 2...")
            elif "Generation 3/" in output:
                progress_bar.progress(0.8)
                status_text.text("Running generation 3...")
            elif "Evolution complete" in output:
                progress_bar.progress(1.0)
                status_text.text("Evolution complete!")

        builtins.print = custom_print

        try:
            # Run evolution
            best_program = alpha_evolve.evolve()

            # Display results
            st.markdown('<div class="sub-header">Best Solution</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="code-box">{best_program.code}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sub-header">Performance</div>', unsafe_allow_html=True)
            st.metric("Fitness Score", f"{best_program.fitness:.2f}/100")

        except Exception as e:
            st.error(f"Error during evolution: {str(e)}")
        finally:
            # Restore original print function
            builtins.print = original_print

# Information about AlphaEvolve
with st.expander("About AlphaEvolve"):
    st.markdown("""
    ### AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery

    AlphaEvolve is a novel evolutionary coding agent developed by Google DeepMind. It enhances the capabilities of large language models (LLMs) in tackling complex scientific and computational problems by orchestrating an autonomous pipeline of LLMs to iteratively improve algorithms.

    #### Key Features:

    1. **Evolutionary Approach**: Uses an evolutionary approach to iteratively improve algorithms, leveraging feedback from evaluators.

    2. **Broad Applicability**: Successfully applied to a range of computational problems, demonstrating versatility.

    3. **Optimization Capabilities**: Optimizes critical components of computational infrastructure, including data center scheduling and hardware accelerator design.

    4. **Novel Algorithm Discovery**: Discovers novel, provably correct algorithms that surpass state-of-the-art solutions.

    This demonstration provides a simplified version of the AlphaEvolve methodology, focusing on the core evolutionary process using LLMs for code generation, evaluation, and improvement.
    """)

if __name__ == "__main__":
    pass
