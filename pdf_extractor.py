import PyPDF2
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

def analyze_paper(pdf_text):
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
        result = summary_chain.run(paper_text=pdf_text)
        return result
    except Exception as e:
        return f"Error analyzing paper: {str(e)}"

def main():
    """Main function to extract and analyze the PDF."""
    pdf_path = "D:\\OpenSourceLibrary\\AlphaEvolve\\AlphaEvolve.pdf"
    
    if os.path.exists(pdf_path):
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Save the extracted text to a file
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(pdf_text)
        
        print(f"Text extracted from PDF and saved to extracted_text.txt")
        
        # Analyze the paper
        analysis = analyze_paper(pdf_text)
        
        # Save the analysis to a file
        with open("paper_analysis.md", "w", encoding="utf-8") as f:
            f.write(analysis)
        
        print(f"Paper analysis completed and saved to paper_analysis.md")
        return pdf_text, analysis
    else:
        print(f"PDF file not found at {pdf_path}")
        return None, None

if __name__ == "__main__":
    main()
