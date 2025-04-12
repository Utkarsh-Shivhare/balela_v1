from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum, auto
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from instruction.writing_agents_templates import (GRAMMER_AND_LANGUAGE_SUPPORT_AGENT_TEMPLATE,
                                                    REPHRASING_AGENT_TEMPLATE,
                                                    ASSIGNMENT_FEEDBACK_AGENT_TEMPLATE,
                                                    CUSTOM_FEEDBACK_AGENT_TEMPLATE,
                                                    STRUCTURAL_FLOW_ANALYSIS_AGENT_TEMPLATE,
                                                    CONTENT_RELEVANCE_CHECK_TEMPLATE,
                                                    CALL_ENCOURAGING_REVISIONS_TEMPLATE,
                                                    RESEARCH_PROJECT_ASSISTANT_TEMPLATE,
                                                    AUTO_COMPLETE_AI_SUGGESTION)
import os
from dotenv import load_dotenv
import logging  
from typing import List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# class GrammarCorrectionResponse(BaseModel):
#     corrected_text: str = Field(description="The text with improved grammar, punctuation, and sentence structure.")
class GrammarErrorDetail(BaseModel):
    error: str = Field(description="The incorrect word or phrase")
    correction: str = Field(description="The correct word or phrase")
    context: str = Field(description="Concise context around the error(including the error) (few words before and after )")
    corrected_sentence: str = Field(description="Full sentence with the correction applied")
    type: str = Field(description="Type of error (spelling/grammar)")
    explanation: str = Field(description="Detailed explanation of the error")

class GrammarCorrectionResponse(BaseModel):
    errors: List[GrammarErrorDetail] = Field(description="List of grammar and language errors with their corrections")



class RephrasingResponse(BaseModel):
    rephrased_text: str = Field(description="The text rephrased according to the specified style or context.")

class StrengthsResponse(BaseModel):
    relevance: str = Field(description="The relevance of the content to the assignment.")
    clarity: str = Field(description="The clarity of the content.")
    conciseness: str = Field(description="The conciseness of the content.")

class WeaknessesResponse(BaseModel):
    structure: str = Field(description="The structure of the content.")
    argumentation: str = Field(description="The argumentation of the content.")
    depth_of_the_content: str = Field(description="The depth of the content.")

class SuggestionsForImprovementResponse(BaseModel):
    add_a_clear_structure: str = Field(description="Suggestions for adding a clear structure.")
    enhance_argumentation: str = Field(description="Suggestions for enhancing argumentation.")
    include_examples_and_data: str = Field(description="Suggestions for including examples and data.")
    conclude_effectively: str = Field(description="Suggestions for concluding effectively.")

# class EssayFeedbackResponse(BaseModel):
#     score: int = Field(description="Score the feedback summary on the essay or assignment.")
#     strengths: List[StrengthsResponse] = Field(description="The strengths of the content.")
#     weaknesses: List[WeaknessesResponse] = Field(description="The weaknesses of the content.")
#     suggestions_for_improvement: List[SuggestionsForImprovementResponse] = Field(description="The suggestions for improvement.")
class SectionFeedback(BaseModel):
    """
    Represents feedback for a specific section of the essay.
    """
    name: str
    score: int = Field(
        ge=0, le=100,
        description="Score of the section out of 100"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Positive aspects of the section"
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="Improvements needed for the section"
    )

class AdditionalAnalysis(BaseModel):
    """
    Represents additional analyses (e.g. Overall Structure and Flow, Language and Style).
    """
    name: str
    score: int = Field(
        ge=0, le=100,
        description="Score out of 100"
    )
    summary: str = Field(
        description="A brief summary of the analysis"
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="Areas needing improvement for this aspect"
    )
class QuestionFeedback(BaseModel):
    question: str = Field(description="The question provided by the user related to the essay.")
    score: int = Field(description="Score out of 100 for how well the question is answered.", ge=0, le=100)
    answered: str = Field(description="Indicates the status of the question's answer (e.g., 'answered', 'partially answered', 'not answered') with concise details")
    where: str = Field(description="Indicates where in the assignment the question is addressed.")
    strengths: List[str] = Field(default_factory=list, description="Strengths in the answer to the question.")
    improvements: List[str] = Field(default_factory=list, description="Suggestions for improving the answer to the question.")


class EssayFeedbackResponse(BaseModel):
    """
    Comprehensive feedback model for the essay.
    """
    overall_rating: int = Field(
        ge=0, le=100,
        description="Overall rating of the essay (0-100)"
    )
    overall_summary: str = Field(
        description="A brief summary of the essay feedback"
    )
    sections: List[SectionFeedback] = Field(
        default_factory=list,
        description="Detailed feedback on individual sections of the assigment."
    )
    additional_analyses: List[AdditionalAnalysis] = Field(
        default_factory=list,
        description="Additional analyses (e.g. structure, style) "
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Final actionable recommendations"
    )
    questions: List[QuestionFeedback] = Field(
        default_factory=list,
        description="List of questions provided by the user, with feedback on whether they are adequately addressed in the assignment text."
    )


class Feedback_Agent_With_Strictness_Level(BaseModel):
    ai_comment: str = Field(description="Comprehensive feedback on the essay or assignment, including strengths, weaknesses, suggestions, and improvement comments, based on strictness level.")


class StructuralFlowAnalysisResponse(BaseModel):
    ai_comment: str = Field(description="Analysis of the structure and flow of the content, including suggestions for improvement.")
    modified_content: str = Field(description="Content modified based on the analysis provided, formatted as per the analysis.")

class ContentRelevanceCheckResponse(BaseModel):
    relevance_comment_summary: str = Field(description="Feedback on the relevance of the content, including suggestions for improvement.")

class RevisionEncouragementResponse(BaseModel):
    ai_feedback: str = Field(description="Feedback encouraging revisions and refinements, including specific suggestions for improvement.")

class ResearchProjectAssistanceResponse(BaseModel):
    ai_feedback: str = Field(description="Feedback on the research project structure, including suggestions for improvement.")
   
class Ai_Auto_Complete_Suggestion(BaseModel):
    ai_suggestion: str = Field(description="A single, contextually relevant and natural autocomplete suggestion that completes the user's input text")
   

class WritingAssistant:
    def __init__(self, OPENAI_API_KEY):
        logging.info("Initializing Writing Assistant.")
        self.llm = self._initialize_llm(OPENAI_API_KEY)

    def _initialize_llm(self, OPENAI_API_KEY):
        logging.info("Initializing LLM with GPT-4o model.")
        return ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.4,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=OPENAI_API_KEY,
        )

    def call_grammar_and_language_support_agent(self, user_input):
        logging.info("Calling grammar and language support agent.")
        parser = JsonOutputParser(pydantic_object=GrammarCorrectionResponse)
        template = GRAMMER_AND_LANGUAGE_SUPPORT_AGENT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": user_input})
        logging.info("Grammar and language support completed successfully.")
        return result

    def call_rephrasing_agent(self, input_text, query=None):
        logging.info("Calling rephrasing agent.")
        parser = JsonOutputParser(pydantic_object=RephrasingResponse)
        # Default rephrasing style if no query is provided
        rephrasing_style = query if query else "clearer and more formal or creative, depending on the context"
        template = REPHRASING_AGENT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "rephrasing_style"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": input_text, "rephrasing_style": rephrasing_style})
        logging.info("Rephrasing completed successfully.")
        return result

    def call_assignment_feedback_agent(self, assignment_text, title, questions):
        logging.info("Calling assignment feedback agent.")
        parser = JsonOutputParser(pydantic_object=EssayFeedbackResponse)
        template = ASSIGNMENT_FEEDBACK_AGENT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input","title","list_of_questions"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": assignment_text, "title": title,"list_of_questions":questions})
        logging.info("Assignment feedback generated successfully.")
        return result
    
    def call_custom_feedback_agent_with_strictness_level(self, input_content_text, strictness_level):
        """Provide feedback on the assignment with customizable strictness levels."""
        logging.info(f"Calling custom feedback agent with strictness level: {strictness_level}.")  
        parser = JsonOutputParser(pydantic_object=Feedback_Agent_With_Strictness_Level)
    
        template = CUSTOM_FEEDBACK_AGENT_TEMPLATE
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input","strictness_level"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": input_content_text,"strictness_level":strictness_level})
        logging.info("Custom feedback generated successfully.")
        return result
    
    def call_structural_flow_analysis(self, user_content: str):
        """Analyze the structure and flow of the provided content."""
        logging.info("Calling structural flow analysis agent.")
        parser = JsonOutputParser(pydantic_object=StructuralFlowAnalysisResponse)

        template = STRUCTURAL_FLOW_ANALYSIS_AGENT_TEMPLATE

        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Invoke the LLM with the prompt
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": user_content})

        logging.info("Structural flow analysis completed successfully.")
        return result
    
    def call_content_relevance_check(self, user_content: str):
        """Check the relevance of the provided content."""
        logging.info("Calling content relevance check agent.")  
        parser = JsonOutputParser(pydantic_object=ContentRelevanceCheckResponse)

        template = CONTENT_RELEVANCE_CHECK_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Invoke the LLM with the prompt
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": user_content})

        logging.info("Content relevance check completed successfully.")
        return result
    
    def call_encouraging_revisions(self, user_content: str):
        """Encourage revisions and refinements in the provided content."""
        logging.info("Calling encouraging revisions agent.") 
        parser = JsonOutputParser(pydantic_object=RevisionEncouragementResponse)

        # Define the prompt for the LLM
        template = CALL_ENCOURAGING_REVISIONS_TEMPLATE

        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Invoke the LLM with the prompt
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": user_content})

        logging.info("Encouraging revisions completed successfully.")
        return result
    
    def call_research_project_assistance(self, user_content: str):
        """Assist with structuring research projects, ensuring all essential elements are included."""
        logging.info("Calling research project assistance agent.")
        parser = JsonOutputParser(pydantic_object=ResearchProjectAssistanceResponse)

        template = RESEARCH_PROJECT_ASSISTANT_TEMPLATE

        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Invoke the LLM with the prompt
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": user_content})

        logging.info("Research project assistance completed successfully.")  
        return result

    def call_suggestion_auto_complete_agent(self, input_text: str):
        """
        Generate auto-complete suggestions using the AI assistant.
        """
        logging.info("Calling auto-complete agent.")
        # Define your prompt template for auto-completion
        parser = JsonOutputParser(pydantic_object=Ai_Auto_Complete_Suggestion)
        template = AUTO_COMPLETE_AI_SUGGESTION
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = prompt | self.llm | parser 
        result = chain.invoke({"input": input_text})
        
        logging.info("Auto-complete suggestions generated successfully.")  
        return result  