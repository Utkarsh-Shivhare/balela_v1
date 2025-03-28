# GRAMMER_AND_LANGUAGE_SUPPORT_AGENT_TEMPLATE = """
#             You are a grammar and language supporter, your task is to improve the following text by ensuring proper grammar, punctuation, and sentence structure.
            
#             You have to use better vocabulary in response.

#             Original Text: {input}

#             {format_instructions}
#         """


GRAMMER_AND_LANGUAGE_SUPPORT_AGENT_TEMPLATE = """
You are an advanced grammar and language correction assistant. Analyze the following text and identify all grammatical, spelling, and language usage errors.

For each error, provide:
1. The specific error found
2. The correct word/phrase
3. A concise context (few words before and after the error including the error)
4. The full corrected sentence
5. The type of error (spelling/grammar/punctuation)
6. A detailed explanation of why it's an error and how to avoid it in the future

Guidelines for Context:
- Provide a short, focused context around the error
- Include 3-5 words before and after the error if present
- Focus on the immediate linguistic environment of the mistake
- Ensure the context helps understand the error without being overly verbose

Original Text: {input}

Please provide a comprehensive analysis following these guidelines:
- Identify multiple errors if present
- Be specific and constructive in your feedback
- Explain the reasoning behind each correction

{format_instructions}
"""

REPHRASING_AGENT_TEMPLATE = """
            Rephrase the following text to make it {rephrasing_style}. Ensure the rephrased text is clear and maintains the original meaning.
            
            Original Text: {input}

            {format_instructions}
        """

# ASSIGNMENT_FEEDBACK_AGENT_TEMPLATE = """
#             Provide detailed feedback on the following assignment. Evaluate it based on criteria such as structure, relevance, argumentation, and grammar.
#             Include a score out of 100, and summarize the strengths and weaknesses in the feedback.
#             Suggest specific improvements, such as additional content, better structure, or enhanced clarity.
#             If the assignment is an essay, ensure it follows a logical structure with a clear introduction, body, and conclusion.
#             If it is a story, comment on narrative elements like plot, character development, and setting.
#             Assignment Text: {input}

#             {format_instructions}
#         """

# ASSIGNMENT_FEEDBACK_AGENT_TEMPLATE = """
#     Provide detailed feedback on the following assignment titled "{title}". Evaluate it based on criteria such as structure, relevance, argumentation, and grammar.
#     Include a score out of 100, and summarize the strengths and weaknesses in the feedback.
#     Suggest specific improvements, such as additional content, better structure, or enhanced clarity.
#     If the assignment is an essay, ensure it follows a logical structure with a clear introduction, body, and conclusion.
#     If it is a story, comment on narrative elements like plot, character development, and setting.

#     Additionally, assess whether the content of the assignment is relevant to the title. 
#     If there are discrepancies, provide specific comments on how the content could better align with the title.

#     Assignment Title: {title}
#     Assignment Text: {input}

#     {format_instructions}
# """


# ASSIGNMENT_FEEDBACK_AGENT_TEMPLATE = """
# Provide a detailed evaluation of the assignment titled "{title}" using the following criteria:

# 1. **Strengths**:
#    - Assess **relevance**, **clarity**, and **conciseness**.
#    - If no feedback is needed for a specific parameter, return `"null"` (e.g., `"relevance": "null"`).


# 2. **Weaknesses**:
#    - Evaluate **structure**, **argumentation**, and **depth_of_the_content**.
#    - If no feedback is needed for a specific parameter, return `"null"` (e.g., `"structure": "null"`).


# 3. **Suggestions for Improvement**:
#    - Recommend specific actions such as adding structure, enhancing arguments, including examples or data, or improving the conclusion.
#    - If no feedback is needed for a specific parameter, return `"null"` (e.g., `"add_a_clear_structure": "null"`).


# Additional Instructions:
# - Provide a score out of 100 based on the evaluation criteria.
# - If the content is not relevant to the title, assign a score of **0**. Clearly explain why the content does not align with the title and provide specific recommendations to improve alignment.
# - For any criteria where no changes are needed, explicitly return `"null"` for each parameter (e.g., `"structure": "null"`, `"relevance": "null"`).

# Input:
# - Assignment Title: {title}
# - Assignment Text: {input}

# {format_instructions}
# """
ASSIGNMENT_FEEDBACK_AGENT_TEMPLATE="""
Comprehensive Assignment Evaluation Framework

Objective: Conduct a thorough, multi-dimensional analysis of the submitted assignment.

Assignment Details:
- Title: "{title}"
- Content Focus: {input}
- List of Questions:{list_of_questions}

Evaluation Dimensions:
You are an expert essay reviewer. Provide the feedback in the following JSON structure:

STRICTLY NOTE:1.Individual SectionFeedback score should not be greater than 100.
              
Output Format Requirements:
{format_instructions}

Guiding Principle: Deliver a comprehensive, objective, and constructive evaluation that supports academic growth and understanding.
"""

CUSTOM_FEEDBACK_AGENT_TEMPLATE = """
            Provide detailed feedback on the following user provided content based on the strictness level: {strictness_level}. 
            Evaluate it based on criteria such as structure, relevance, argumentation, and grammar.
            Summarize the strengths and weaknesses in the feedback.
            Suggest specific improvements, such as additional content, better structure, or enhanced clarity.
            Comment on how much the user can improve based on the analysis of the content.
            For example, you might say: "Your argument is valid, but your evidence is vague. Ensure every claim is supported by specific examples."
            User Content Text: {input}

            {format_instructions}
        """

STRUCTURAL_FLOW_ANALYSIS_AGENT_TEMPLATE = """
        Analyze the following content for its structure and flow. 
        Evaluate coherence and flow, suggesting improvements for smooth transitions between paragraphs. 
    
        You analysis comment should be more specific to the content.
        For example, you might say: "Paragraph two repeats ideas from paragraph one. Consider expanding on the consequences of Macbeth’s ambition in paragraph three."
        Content:
        {input}

        Please provide your analysis and modified content.
        

        {format_instructions}
        """

CONTENT_RELEVANCE_CHECK_TEMPLATE = """
        Analyze the following content strictly for relevance to the original topic or task.

        Evaluation Criteria:
        - Identify off-topic digressions
        - Detect unnecessary or redundant information
        - Flag content that does not directly contribute to the core objective
        - Assess whether each section meaningfully advances the discussion

        Specific Reporting Requirements:
        - For each relevance issue, provide:
        1. Exact paragraph number
        2. Exact sentence number within that paragraph
        3. The full sentence text
        4. A clear explanation of why it's irrelevant

        Detailed Analysis summary format:
        - If off-topic: "Paragraph X, Sentence Y: [Full Sentence] - This content deviates from the main topic because..."
        - If redundant: "Paragraph X, Sentence Y: [Full Sentence] - This repeats information previously stated in..."
        - If tangential: "Paragraph X, Sentence Y: [Full Sentence] - This information does not contribute directly to the core discussion..."

        Content Under Review:
        {input}

        Provide a comprehensive relevance check with precise location references and clear explanations.
        
        {format_instructions}
        """

CALL_ENCOURAGING_REVISIONS_TEMPLATE = """
        Analyze the following content and encourage the learner to improve weak areas iteratively. 
        Provide constructive feedback and specific suggestions for revisions.

        For example, you might say: "This essay is strong overall, but the introduction lacks a hook. Let’s try to make it more engaging!"

        Content:
        {input}

        Please provide your encouragement comment and any modified content.
        
        {format_instructions}
        """


RESEARCH_PROJECT_ASSISTANT_TEMPLATE = """
        Analyze the following research project content and ensure all essential elements are included, such as a clear hypothesis, methodology, and findings.
        Provide constructive feedback and specific suggestions for improvement.

        For example, you might say: "Your research lacks a defined methodology." e.t.c

        Content:
        {input}

        Please provide your feedback and any modified content.
        
        {format_instructions}
        """
AUTO_COMPLETE_AI_SUGGESTION = """Generate a single, most likely autocomplete suggestion that naturally and contextually completes the input text: {input}.

please provide the response in this format.
{format_instructions}
"""