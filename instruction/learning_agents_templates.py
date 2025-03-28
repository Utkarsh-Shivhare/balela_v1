QUESTION_ANSWER_GENERATOR_TEMPLATE = """
            Based on the following content, generate a list of insightful questions that can help someone understand the material better. 
            Focus on key concepts, missing details, and implications. Return the questions in JSON format as per the specified structure:
            If you don't have any content to analyze then simply return an empty list.
            {format_instructions}
            
            Content to analyze: {input}
        """

ANALYZE_UPLOADED_MATERIAL_CONTENT_TEMPLATE = """
            Provide a comprehensive analysis of the given content. Your analysis should:
            1. Identify and highlight the most significant themes, key characters, or critical events
            2. Break down complex ideas into easily digestible insights
            3. Offer context and deeper understanding of the material
            4. Capture the essence of the content in a structured, clear manner

            If no content is provided, return an empty list.

            {format_instructions}

            Content to analyze: {input}
        """
GUIDED_QUESTION_ASSISTANT_TEMPLATE = """
            Based on the content provided, offer hints or clues to help the user solve the following question:
            {question}
            Consider the user's current input answer: {user_input_answer}.
            Provide guidance on whether the user is on the right track and suggest the next steps without giving away the answer.
            {format_instructions}

            Content to analyze: {input}
        """

DETAILED_ANSWER_AGENT_TEMPLATE = """
            Using the provided content, generate a detailed and direct answer to the following question:
            {question}
            Ensure the answer is comprehensive and directly addresses the question using the context provided.
            {format_instructions}

            Content to analyze: {input}
        """

ANALYZE_TEST_TEMPLATE = """
            Provide a summary feedback on the learner's performance across all question-answer pairs using the STAR methodology.
            Focus on describing the context of the questions, the learning objectives, the evaluation of the learner's responses, and the overall performance with suggestions for improvement.

            Additionally, based on the uploaded context, offer specific learning suggestions or areas to focus on that could enhance the learner's understanding and performance in future assessments.
            For example, if a question is incorrectly answered and relates to variables or data types, suggest reviewing these topics in the uploaded context if they are covered.
            Mention the topic names explicitly, such as "Variables", "Data Types", "Control Structures", etc., if they are relevant to the learner's areas of improvement.
            Address the user directly in your response, using phrases like "You can improve on..." or "Consider focusing on...".
            
            {format_instructions}

            Context: {context}
            Question-Answer Pairs: {qa_pairs}
        """
