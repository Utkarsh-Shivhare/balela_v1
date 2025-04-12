from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from instruction.learning_agents_templates import (QUESTION_ANSWER_GENERATOR_TEMPLATE,
                                                ANALYZE_UPLOADED_MATERIAL_CONTENT_TEMPLATE,
                                                GUIDED_QUESTION_ASSISTANT_TEMPLATE,
                                                DETAILED_ANSWER_AGENT_TEMPLATE,
                                                ANALYZE_TEST_TEMPLATE)
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import logging
import numpy as np
from database import init_db, save_document_to_db, get_document_from_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Questions(BaseModel):
    questions: List[str] = Field(description="A list of insightful questions generated from the input content.")

class AnalyzedData(BaseModel):
    analyzed_response: str = Field(description="Comprehensive analysis capturing key insights, themes, and critical elements of the input content.")

class GuidedQuestionResponse(BaseModel):
    hints: str = Field(description="Hints or clues to guide the user based on their current input and the content source.")

class DetailedAnswerResponse(BaseModel):
    answer: str = Field(description="A detailed and direct answer to the user's question based on the provided context.")

class ImprovementAnalysisResponse(BaseModel):
    summary_feedback: str = Field(description="Feedback on where the learner can improve based on their answers.")

class LearningAssistant:
    def __init__(self, OPENAI_API_KEY):
        logging.info("Initializing Learning Assistant.")
        self.llm = self._initialize_llm(OPENAI_API_KEY)
        self.embeddings = self._initialize_embeddings(OPENAI_API_KEY)
        # Initialize database connection
        self.engine, self.Session = init_db()

    def _initialize_llm(self, OPENAI_API_KEY):
        logging.info("Initializing LLM with GPT-4o model.")
        return ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=OPENAI_API_KEY,
        )
    
    def _initialize_embeddings(self, OPENAI_API_KEY):
        logging.info("Initializing OpenAI Embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query, user_id, document_id):
        logging.info(f"Searching for query: '{query}' for user_id: '{user_id}' and document_id: '{document_id}'")
        try:
            # Get embeddings for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Create database session
            session = self.Session()
            
            # Get document from database
            doc = get_document_from_db(session, user_id, document_id)
            session.close()
            
            if not doc:
                logging.warning(f"No documents found for user_id: {user_id} and document_id: {document_id}")
                return []
            
            results = []
            if doc["is_book"]:  # Only process if it's marked as a book
                similarity = self.cosine_similarity(query_embedding, doc["embedding"])
                results.append({
                    "content": doc["content"],
                    "score": similarity
                })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:15]  # Return top 15 results
            
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []

    def format_sources(self, search_results):
        logging.info("Formatting search results.")
        if not search_results:
            return "No relevant content found."
            
        formatted_sources = "=================\n".join([
            f'CONTENT: {result["content"]}' for result in search_results
        ])
        logging.debug(f"Formatted sources length: {len(formatted_sources)} characters")
        return formatted_sources

    def _get_formatted_sources(self, user_id, document_id, question):
        logging.info(f"Getting formatted sources for user_id: '{user_id}', document_id: '{document_id}', question: '{question}'")
        search_results = self.search(question, user_id, document_id)
        return self.format_sources(search_results)

    def generate_questions_agent(self, sources_formatted):
        logging.info("Generating questions from sources.")
        parser = JsonOutputParser(pydantic_object=Questions)
        template = QUESTION_ANSWER_GENERATOR_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": sources_formatted})
        logging.info("Questions generated successfully.")
        return result

    def analyze_uploaded_material_content(self, sources_formatted):
        logging.info("Analyzing uploaded material content.")
        parser = JsonOutputParser(pydantic_object=AnalyzedData)
        template = ANALYZE_UPLOADED_MATERIAL_CONTENT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": sources_formatted})
        logging.info("Content analysis completed successfully.")
        return result

    def guided_question_assistant(self, sources_formatted, question, user_input_answer):
        logging.info("Providing guided question assistance.")
        parser = JsonOutputParser(pydantic_object=GuidedQuestionResponse)
        template = GUIDED_QUESTION_ASSISTANT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "question", "user_input_answer"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": sources_formatted, "question": question, "user_input_answer": user_input_answer})
        logging.info("Guided question assistance provided successfully.")
        return result

    def detailed_answer_agent(self, sources_formatted, question):
        logging.info("Generating detailed answer.")
        parser = JsonOutputParser(pydantic_object=DetailedAnswerResponse)
        template = DETAILED_ANSWER_AGENT_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        result = chain.invoke({"input": sources_formatted, "question": question})
        logging.info("Detailed answer generated successfully.")
        return result

    def analyze_test(self, combined_context, qa_pairs):
        logging.info("Analyzing test performance.")
        parser = JsonOutputParser(pydantic_object=ImprovementAnalysisResponse)
        template = ANALYZE_TEST_TEMPLATE
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "qa_pairs"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser

        result = chain.invoke({
            "context": combined_context,
            "qa_pairs": qa_pairs
        })
        logging.info("Test performance analysis completed successfully.")
        return result

    def call_generate_questions_agent(self, user_id, document_id):
        logging.info(f"Calling generate questions agent for user_id: '{user_id}', document_id: '{document_id}'")
        sources_formatted = self._get_formatted_sources(user_id, document_id, "*")
        return self.generate_questions_agent(sources_formatted)

    def call_analyze_content_agent(self, user_id, document_id):
        logging.info(f"Calling analyze content agent for user_id: '{user_id}', document_id: '{document_id}'")
        sources_formatted = self._get_formatted_sources(user_id, document_id, "*")
        return self.analyze_uploaded_material_content(sources_formatted)

    def call_guided_question_assistant(self, user_id, document_id, question, user_input_answer):
        logging.info(f"Calling guided question assistant for user_id: '{user_id}', document_id: '{document_id}', question: '{question}'")
        sources_formatted = self._get_formatted_sources(user_id, document_id, question)
        return self.guided_question_assistant(sources_formatted, question, user_input_answer)

    def call_detailed_answer_agent(self, user_id, document_id, question):
        logging.info(f"Calling detailed answer agent for user_id: '{user_id}', document_id: '{document_id}', question: '{question}'")
        sources_formatted = self._get_formatted_sources(user_id, document_id, question)
        return self.detailed_answer_agent(sources_formatted, question)

    def call_analyze_test(self, user_id, document_id, qa_pairs):
        logging.info(f"Calling analyze test for user_id: '{user_id}', document_id: '{document_id}'")
        all_contexts = []
        for qa in qa_pairs:
            question = qa.question
            context = self._get_formatted_sources(user_id, document_id, question)
            all_contexts.append(context)

        combined_context = "\n".join(all_contexts)
        return self.analyze_test(combined_context, qa_pairs)
    
    
    