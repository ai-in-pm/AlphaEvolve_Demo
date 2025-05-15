# AlphaEvolve Paper Demonstration

This project provides an interactive demonstration of the AlphaEvolve paper, allowing users to explore the concepts and methodologies presented in the research.

https://github.com/user-attachments/assets/c37b75da-be06-4819-9033-20695ba636b2

## Features

- **Paper Analysis**: Comprehensive summary and key takeaways from the AlphaEvolve paper
- **Interactive Demo**: Simulated experiments based on the AlphaEvolve methodology
- **AI Agent**: PhD-level AI expert to answer questions about the paper
- **Visualizations**: Performance graphs and evolutionary progress visualization

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd AlphaEvolve
   ```

2. **Create and activate a virtual environment**:
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   source venv/bin/activate     # Linux/Mac
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY="your-openai-api-key-here"
     ```

5. **Run the application**:
   ```
   streamlit run app.py
   ```

## Components

- `app.py`: Main Streamlit application
- `pdf_extractor.py`: Utility to extract and analyze PDF content
- `.env`: Environment variables configuration
- `requirements.txt`: Project dependencies

## Usage

1. **Paper Summary**: View a comprehensive academic summary of the AlphaEvolve paper
2. **Key Takeaways**: Explore the main contributions and limitations of the research
3. **Interactive Demo**: Run simulated experiments based on the AlphaEvolve methodology
4. **AI Agent Q&A**: Ask technical questions about the paper and get expert responses

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## Note

This demonstration is for educational purposes and provides a simplified simulation of the AlphaEvolve methodology. For the full implementation details, please refer to the original paper and code repository.
