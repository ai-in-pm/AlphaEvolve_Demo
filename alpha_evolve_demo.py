import random
import numpy as np
import time
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Program:
    """Represents a candidate program in the AlphaEvolve system."""
    
    def __init__(self, code, description=""):
        self.code = code
        self.description = description
        self.fitness = None
        self.error = None
    
    def __str__(self):
        return f"Program: {self.description}\nFitness: {self.fitness}\n{self.code}"


class AlphaEvolveDemo:
    """A simplified demonstration of the AlphaEvolve methodology."""
    
    def __init__(self, problem_description, population_size=10, generations=5, model="gpt-4o"):
        self.problem_description = problem_description
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_program = None
        self.llm = ChatOpenAI(temperature=0.7, model=model)
    
    def initialize_population(self):
        """Generate initial population of programs using LLM."""
        print("Initializing population...")
        
        # Create a prompt template for generating initial programs
        init_template = PromptTemplate(
            input_variables=["problem"],
            template="""
            You are an expert programmer tasked with creating a solution for the following problem:
            
            {problem}
            
            Generate a Python function that solves this problem. The function should be:
            1. Efficient
            2. Well-commented
            3. Ready to execute
            
            Provide only the Python code without any explanations or markdown formatting.
            """
        )
        
        # Create the chain
        init_chain = LLMChain(llm=self.llm, prompt=init_template)
        
        # Generate initial population
        for i in range(self.population_size):
            try:
                code = init_chain.run(problem=self.problem_description)
                program = Program(code, f"Initial Program {i+1}")
                self.population.append(program)
                print(f"Generated program {i+1}/{self.population_size}")
            except Exception as e:
                print(f"Error generating program: {str(e)}")
        
        return self.population
    
    def evaluate_fitness(self, program):
        """Evaluate the fitness of a program using LLM."""
        # Create a prompt template for evaluating programs
        eval_template = PromptTemplate(
            input_variables=["problem", "code"],
            template="""
            You are an expert code evaluator. Assess the following Python code that attempts to solve this problem:
            
            PROBLEM:
            {problem}
            
            CODE:
            {code}
            
            Evaluate this code on a scale from 0 to 100 based on:
            1. Correctness (does it solve the problem?)
            2. Efficiency (time and space complexity)
            3. Readability and maintainability
            4. Error handling
            
            Return only a single number between 0 and 100 representing your evaluation.
            """
        )
        
        # Create the chain
        eval_chain = LLMChain(llm=self.llm, prompt=eval_template)
        
        try:
            # Get fitness score
            result = eval_chain.run(problem=self.problem_description, code=program.code)
            
            # Extract numeric score
            fitness = float(result.strip())
            program.fitness = fitness
            program.error = None
            return fitness
        except Exception as e:
            program.fitness = 0
            program.error = str(e)
            return 0
    
    def select_parents(self, tournament_size=3):
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(self.population_size):
            # Select random candidates for tournament
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # Select the best from the tournament
            winner = max(tournament, key=lambda p: p.fitness if p.fitness is not None else 0)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parent programs using LLM."""
        # Create a prompt template for crossover
        crossover_template = PromptTemplate(
            input_variables=["problem", "parent1", "parent2"],
            template="""
            You are an expert programmer tasked with combining the best aspects of two programs.
            
            PROBLEM:
            {problem}
            
            PROGRAM 1:
            {parent1}
            
            PROGRAM 2:
            {parent2}
            
            Create a new program that combines the strengths of both programs while addressing their weaknesses.
            Provide only the Python code without any explanations or markdown formatting.
            """
        )
        
        # Create the chain
        crossover_chain = LLMChain(llm=self.llm, prompt=crossover_template)
        
        try:
            # Generate child program
            child_code = crossover_chain.run(
                problem=self.problem_description,
                parent1=parent1.code,
                parent2=parent2.code
            )
            
            return Program(child_code, "Crossover Program")
        except Exception as e:
            print(f"Error in crossover: {str(e)}")
            # Return a copy of the better parent if crossover fails
            return Program(
                parent1.code if parent1.fitness > parent2.fitness else parent2.code,
                "Fallback Crossover Program"
            )
    
    def mutate(self, program):
        """Mutate a program using LLM."""
        # Create a prompt template for mutation
        mutation_template = PromptTemplate(
            input_variables=["problem", "code"],
            template="""
            You are an expert programmer tasked with improving the following code:
            
            PROBLEM:
            {problem}
            
            CURRENT CODE:
            {code}
            
            Improve this code by making ONE significant change that could enhance its:
            1. Efficiency
            2. Readability
            3. Correctness
            4. Error handling
            
            Provide only the improved Python code without any explanations or markdown formatting.
            """
        )
        
        # Create the chain
        mutation_chain = LLMChain(llm=self.llm, prompt=mutation_template)
        
        try:
            # Generate mutated program
            mutated_code = mutation_chain.run(
                problem=self.problem_description,
                code=program.code
            )
            
            return Program(mutated_code, "Mutated Program")
        except Exception as e:
            print(f"Error in mutation: {str(e)}")
            # Return a copy of the original program if mutation fails
            return Program(program.code, "Fallback Mutation Program")
    
    def evolve(self):
        """Run the evolutionary process."""
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        print("\nEvaluating initial population...")
        for i, program in enumerate(self.population):
            fitness = self.evaluate_fitness(program)
            print(f"Program {i+1} fitness: {fitness}")
        
        # Track best program
        self.best_program = max(self.population, key=lambda p: p.fitness if p.fitness is not None else 0)
        print(f"\nInitial best fitness: {self.best_program.fitness}")
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n--- Generation {generation+1}/{self.generations} ---")
            
            # Select parents
            parents = self.select_parents()
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best program
            new_population.append(self.best_program)
            
            # Fill the rest of the population with crossover and mutation
            while len(new_population) < self.population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover or mutation
                if random.random() < 0.7:  # 70% chance of crossover
                    child = self.crossover(parent1, parent2)
                else:  # 30% chance of mutation
                    parent = random.choice([parent1, parent2])
                    child = self.mutate(parent)
                
                new_population.append(child)
            
            # Update population
            self.population = new_population
            
            # Evaluate new population
            print("Evaluating new population...")
            for i, program in enumerate(self.population):
                if program.fitness is None:  # Only evaluate programs without fitness
                    fitness = self.evaluate_fitness(program)
                    print(f"Program {i+1} fitness: {fitness}")
            
            # Update best program
            current_best = max(self.population, key=lambda p: p.fitness if p.fitness is not None else 0)
            if current_best.fitness > self.best_program.fitness:
                self.best_program = current_best
                print(f"New best fitness: {self.best_program.fitness}")
            else:
                print(f"Best fitness remains: {self.best_program.fitness}")
        
        print("\n--- Evolution complete ---")
        print(f"Best program fitness: {self.best_program.fitness}")
        return self.best_program


# Example usage
if __name__ == "__main__":
    # Define a problem
    problem = """
    Create a function to find the longest palindromic substring in a given string.
    A palindrome is a string that reads the same backward as forward.
    
    Example:
    Input: "babad"
    Output: "bab" or "aba" (both are valid)
    
    Input: "cbbd"
    Output: "bb"
    """
    
    # Create AlphaEvolve instance
    alpha_evolve = AlphaEvolveDemo(
        problem_description=problem,
        population_size=5,
        generations=3
    )
    
    # Run evolution
    best_program = alpha_evolve.evolve()
    
    # Print best program
    print("\nBest Program:")
    print(best_program.code)
