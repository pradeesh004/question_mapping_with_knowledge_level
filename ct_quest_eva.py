
#importing requiring libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re

# vectors from the vector stores as retriver
def rag():
    loader = PyPDFLoader(r"Q:\projects\ctp\critical_thinking_user answer evaluation\attention-is-all-you-need-Paper.pdf")
    doc = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=500)
    splitted = splitter.split_documents(doc)

    embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")
    vectors = FAISS.from_documents(splitted,embeddings)

    retrivals = vectors.as_retriever()
    return retrivals

#defininf llms
def llm(Model_name):
    llm = ChatGroq(
        model_name = Model_name,
        temperature=0.4,
        api_key="api key"
    )
    return llm

#define question generation prompt
def question_promtp(topic,retriver):

    prompt = ChatPromptTemplate.from_template(
        """
You are a critical thinking question generator. Your task is to analyze the following content and generate exactly 3 questions that test critical thinking ability based STRICTLY on the given content {context} and topic {topic}.

## IMPORTANT CONSTRAINTS
- Generate questions ONLY from the provided content
- Do NOT introduce external information or examples
- Stay strictly within the boundaries of the given topic and content
- Questions must be answerable using the provided content
- Focus on the specific topic given

## CRITICAL THINKING DIMENSIONS GUIDE

Use these **critical thinking dimensions** as guidance to create thought-provoking questions that challenge reasoning and analysis:

### WHO
- Who benefits from this?
- Who is this harmful to?
- Who makes decisions about this?
- Who is most directly affected?
- Who would be the best person to consult?
- Who deserves recognition for this?

### WHAT
- What are the strengths/weaknesses?
- What is another perspective?
- What is another alternative?
- What would be a counter-argument?
- What is the best/worst case scenario?
- What is most/least important?
- What can we do to make a positive change?

### WHERE
- Where would we see this in the real world?
- Where are there similar concepts/situations?
- Where is there the most need for this?
- Where can we get more information?
- Where are the areas for improvement?

### WHEN
- When is this acceptable/unacceptable?
- When would this benefit our society?
- When would this cause a problem?
- When is the best time to take action?
- When will we know we've succeeded?
- When has this played a part in our history?

### WHY
- Why is this a problem/challenge?
- Why is it relevant to me/others?
- Why is this the best/worst scenario?
- Why are people influenced by this?
- Why should people know about this?
- Why has it been this way for so long?

### HOW
- How is this similar to _______?
- How does this disrupt things?
- How do we know the truth about this?
- How does this benefit us/others?
- How does this harm us/others?
- How do we see this in the future?
- How can we change this for our good?

## QUESTION TYPES TO PRIORITIZE

### Analysis Questions:
- Break down components, relationships, causes/effects
- Examine patterns and connections within the content
- Compare and contrast elements from the material

### Evaluation Questions:
- Judge quality, effectiveness, importance, validity
- Assess strengths and weaknesses based on content
- Critique approaches or solutions presented

### Synthesis Questions:
- Connect ideas from different parts of the content
- Predict outcomes based on given information
- Propose solutions using content as foundation

### Application Questions:
- Apply concepts to scenarios within the context
- Use principles from content in new situations
- Transfer learning to related contexts mentioned in material

## QUALITY REQUIREMENTS

Each question should:
- Be **open-ended and thought-provoking**
- Require **analysis, evaluation, or synthesis** of the given information
- Be **answerable using ONLY the provided context**
- Test **different aspects of critical thinking**
- **Adapt the critical thinking dimensions** to fit the context naturally
- **Avoid factual recall or binary yes/no questions**

## STRICT OUTPUT FORMAT

**Question 1:** [Your question here]

**Question 2:** [Your question here]

**Question 3:** [Your question here]

## CONTENT ANALYSIS INSTRUCTIONS

1. **Read the context thoroughly** to understand key concepts, relationships, and implications
2. **Identify the main topic focus** and ensure questions align with it
3. **Look for content elements** that can support higher-order thinking questions
4. **Ensure variety** by targeting different critical thinking dimensions (Who, What, Where, When, Why, How)
5. **Verify content sufficiency** - can each question be fully answered from the given material?

**CRITICAL**: Do not simply copy the dimension questions. **Adapt them** to fit the context naturally and meaningfully. Focus on creating questions that genuinely challenge critical thinking while remaining strictly within the bounds of the provided content {context} and topic {topic}.
        """
    )
    return prompt

# defining evaluation prompt
def evalutaion_prompt(questions,retriver):

    prompts = ChatPromptTemplate.from_template(
        """

You are an expert educational assessor that maps questions to Bloom's Taxonomy knowledge levels based on their cognitive demands and relationship to source content.

You will be given:
- {questions}: The questions to be evaluated
- {context}: The reference content the questions are based on


## KNOWLEDGE LEVEL FRAMEWORK

**K1 (REMEMBER)**: Retrieving knowledge from memory/content
- **Verbs**: name, list, recall, recognize, identify, define, state, what is, who is, when did

**K2 (UNDERSTAND)**: Determining meaning of information  
- **Verbs**: explain, describe, interpret, translate, paraphrase, summarize, classify, why does, how does

**K3 (APPLY)**: Using information in new situations
- **Verbs**: use, employ, apply, implement, execute, demonstrate, solve, calculate, show how

**K4 (ANALYZE)**: Breaking down and examining relationships
- **Verbs**: analyze, compare, contrast, examine, categorize, differentiate, distinguish, what are the differences

**K5 (EVALUATE)**: Making judgments based on criteria
- **Verbs**: evaluate, judge, assess, critique, justify, rank, defend, which is better, argue

**K6 (CREATE)**: Combining elements to form something new
- **Verbs**: create, design, develop, generate, construct, formulate, plan, propose, invent

## EVALUATION INSTRUCTIONS

1. **Analyze the cognitive process** required to answer each question using the given content
2. **Identify the primary verb** that determines the knowledge level
3. **Map to the highest cognitive level** demanded by the question
4. **Consider content relationship**: What must the user do with the source material?

## STRICT OUTPUT FORMAT

For each question, provide ONLY this format:

```
question: [exact question text]
knowledge level: [K1/K2/K3/K4/K5/K6]
verb found: [primary verb(s) from question that determined classification]
```

## CLASSIFICATION RULES

- **K1**: Direct recall/recognition from content (no processing required)
- **K2**: Explaining/interpreting content in own words (comprehension required)
- **K3**: Using content knowledge in new contexts (transfer required)
- **K4**: Examining content parts/relationships (analysis required)
- **K5**: Judging content quality/validity (evaluation required)  
- **K6**: Producing new content using source as foundation (synthesis required)

## EXAMPLES

```
question: What is photosynthesis?
knowledge level: K1
verb found: what is

question: Explain how photosynthesis works in plants.
knowledge level: K2
verb found: explain

question: Compare the effectiveness of photosynthesis in different plant types.
knowledge level: K4
verb found: compare

question: Design an experiment to test photosynthesis rates.
knowledge level: K6
verb found: design
```

## CRITICAL INSTRUCTIONS

- **Use ONLY the specified output format**
- **No additional commentary or explanations**
- **Focus on the primary cognitive demand of each question**
- **Consider what the user must actually DO with the content to answer**
- **Map to highest cognitive level if multiple processes are involved**
- **Analyze each question independently**

Now evaluate the provided questions {questions} against their source content {context} using this framework.
        """
    )
    return prompts

# cleaning the llms's response
def extract_questions_only(response):
    """
    Extract exactly 3 questions from model response with improved pattern matching
    """
    answer = response['answer']

    # Remove <think> tags if present
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)

    # Clean up the response - remove common intro patterns
    intro_patterns = [
        r'^.*?(?=\*\*Question 1:\*\*)',
        r'^.*?(?=Question 1:)',
        r'^.*?(?=1\.)',
        r'Here are.*?questions.*?:?\s*',
        r'Based on.*?:?\s*',
        r'The following.*?:?\s*',
    ]

    for pattern in intro_patterns:
        answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)

    question_blocks = []

    # Method 1: Extract **Question X:** format
    pattern1 = r'\*\*Question (\d+):\*\*\s*(.*?)(?=\n\s*\*\*Question \d+:|\n\s*\*\*|\Z)'
    matches1 = re.findall(pattern1, answer, re.DOTALL)

    if len(matches1) >= 3:
        for num, content in matches1[:3]:
            clean_content = content.strip().replace('\n', ' ').strip()
            question_blocks.append(f"**Question {num}:** {clean_content}")

    # Method 2: Extract Question X: format (without bold)
    elif re.search(r'Question \d+:', answer):
        pattern2 = r'Question (\d+):\s*(.*?)(?=\nQuestion \d+:|\n\n|\Z)'
        matches2 = re.findall(pattern2, answer, re.DOTALL)
        for num, content in matches2[:3]:
            clean_content = content.strip().replace('\n', ' ').strip()
            question_blocks.append(f"**Question {num}:** {clean_content}")

    # Method 3: Extract numbered list format
    elif re.search(r'^\d+\.', answer, re.MULTILINE):
        pattern3 = r'^(\d+)\.\s*(.*?)(?=^\d+\.|\Z)'
        matches3 = re.findall(pattern3, answer, re.DOTALL | re.MULTILINE)
        for num, content in matches3[:3]:
            clean_content = content.strip().replace('\n', ' ').strip()
            question_blocks.append(f"**Question {num}:** {clean_content}")

    # Method 4: Line-by-line parsing (most robust)
    if len(question_blocks) < 3:
        question_blocks = []
        lines = answer.split('\n')
        current_question = ""
        question_num = 0

        for line in lines:
            line = line.strip()

            # Check if line starts a new question
            if (re.match(r'\*\*Question \d+:', line) or
                    re.match(r'Question \d+:', line) or
                    re.match(r'\d+\.', line)):

                # Save previous question if exists
                if current_question and question_num > 0:
                    question_blocks.append(f"**Question {question_num}:** {current_question.strip()}")

                # Start new question
                question_num += 1
                # Extract question text after the numbering
                current_question = re.sub(r'^\*\*Question \d+:\*\*\s*', '', line)
                current_question = re.sub(r'^Question \d+:\s*', '', current_question)
                current_question = re.sub(r'^\d+\.\s*', '', current_question)

            elif line and question_num > 0 and not line.startswith('**') and line != '':
                # Continue previous question if not empty and not a new section
                current_question += " " + line

            # Stop after collecting 3 questions
            if question_num >= 3 and current_question:
                question_blocks.append(f"**Question {question_num}:** {current_question.strip()}")
                break

        # Don't forget the last question if we didn't reach 3 yet
        if current_question and question_num > 0 and len(question_blocks) < 3:
            question_blocks.append(f"**Question {question_num}:** {current_question.strip()}")

    # Ensure we have exactly 3 questions, pad with placeholder if needed
    while len(question_blocks) < 3:
        question_blocks.append(f"**Question {len(question_blocks) + 1}:** [Question not generated properly]")

    # Return only first 3 questions
    return '\n\n'.join(question_blocks[:3])

# Debug function to troubleshoot
def debug_extraction(response, model_name):
    """
    Debug function to see what's happening with question extraction
    """
    answer = response['answer']
    print(f"\n=== DEBUG for {model_name} ===")
    print(f"Raw answer length: {len(answer)}")
    print(f"Raw answer:\n{repr(answer[:500])}...")

    # Count different question patterns
    patterns = {
        "**Question X:**": len(re.findall(r'\*\*Question \d+:', answer)),
        "Question X:": len(re.findall(r'Question \d+:', answer)),
        "Number.": len(re.findall(r'^\d+\.', answer, re.MULTILINE))
    }

    for pattern_name, count in patterns.items():
        print(f"Found {count} matches for pattern '{pattern_name}'")

    print("=== END DEBUG ===\n")

    return extract_questions_only(response)


# Function to clean evaluation responses
def clean_evaluation_response(response):
    """
    Clean the evaluation response to remove thinking tags and extract only the structured evaluation
    """
    answer = response['answer']

    # Remove <think> tags and their content
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)

    # Remove any leading/trailing whitespace
    answer = answer.strip()

    # If the response starts with unwanted text before "Question Quality Analysis", extract from there
    quality_analysis_match = re.search(r'\*\*Question Quality Analysis:\*\*', answer)
    if quality_analysis_match:
        answer = answer[quality_analysis_match.start():]

    # Clean up extra newlines
    answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)

    return answer

# main work
if __name__ == "__main__":

    print("HI, KINDLY ENTER THE TOPIC BELOW:\n")
    topic = input()

    retriver = rag()

    ques_prompt = question_promtp(topic,retriver)

    #question_generation
    llm1 = llm("deepseek-r1-distill-llama-70b")
    llm2 = llm("gemma2-9b-it")
    llm3 = llm("openai/gpt-oss-120b")

    document_chain_1 = create_stuff_documents_chain(llm1,ques_prompt)
    document_chain_2 = create_stuff_documents_chain(llm2,ques_prompt)
    document_chain_3 = create_stuff_documents_chain(llm3,ques_prompt)

    retrival_chain_1 = create_retrieval_chain(retriver,document_chain_1)
    retrival_chain_2 = create_retrieval_chain(retriver,document_chain_2)
    retrival_chain_3 = create_retrieval_chain(retriver,document_chain_3)

    response_llm1 = retrival_chain_1.invoke({"input":topic,"topic":topic})
    response_llm2 = retrival_chain_2.invoke({"input":topic,"topic":topic})
    response_llm3 = retrival_chain_3.invoke({"input":topic,"topic":topic})

    questions_llm1 = extract_questions_only(response_llm1)
    questions_llm2 = extract_questions_only(response_llm2)
    questions_llm3 = extract_questions_only(response_llm3)

    #questions = [questions_llm1,questions_llm2,questions_llm3]

    print("="*40)
    print("\nQUESTIONS GENERATED BY THE MODEL DEEPSEEK....\n")
    print(extract_questions_only(response_llm1))


    print("=" * 40)
    print("\nQUESTIONS GENERATED BY THE MODEL GOOGLE GEMMA 2....\n")
    print(extract_questions_only(response_llm2))


    print("=" * 40)
    print("\nQUESTIONS GENERATED BY THE MODEL OPENAI/OSS....\n")
    print(extract_questions_only(response_llm3))

    # questions evaluation

    ques_prompt_llm1 = evalutaion_prompt(questions_llm1,retriver)
    ques_prompt_llm2 = evalutaion_prompt(questions_llm2,retriver)
    ques_prompt_llm3 = evalutaion_prompt(questions_llm3,retriver)

    #eval_ques_prompts = [ques_prompt_llm1,ques_prompt_llm2,ques_prompt_llm3]

    #eval_model_1

    # eval_document_chains_llm1 = []
    #
    # for ques_prompt in eval_ques_prompts:
    #     eval_doc_chain = create_stuff_documents_chain(llm1, ques_prompt)
    #     eval_document_chains_llm1.append(eval_doc_chain)

    # eval_retrival_chains_llm1 = []
    #
    # for eval_doc_chain in eval_document_chains_llm1:
    #     eval_retrival_chain = create_retrieval_chain(retriver, eval_doc_chain)
    #     eval_retrival_chains_llm1.append(eval_retrival_chain)

    # answers = []
    # i=0
    # for retrival_chain in eval_retrival_chains_llm1:
    #     response_llm1 = retrival_chain.invoke({"input":f"Evaluate these questions:{questions[i]}","questions": questions[i]})
    #     answers.append(response_llm1)

#-----------------------------------------------------------------------------------------------------------------------
    eval_document_chain_1_llm1 = create_stuff_documents_chain(llm1,ques_prompt_llm1)
    eval_document_chain_2_llm1 = create_stuff_documents_chain(llm1,ques_prompt_llm2)
    eval_document_chain_3_llm1 = create_stuff_documents_chain(llm1,ques_prompt_llm3)

    eval_retrival_chain_1_llm1 = create_retrieval_chain(retriver,eval_document_chain_1_llm1)
    eval_retrival_chain_2_llm1 = create_retrieval_chain(retriver,eval_document_chain_2_llm1)
    eval_retrival_chain_3_llm1 = create_retrieval_chain(retriver,eval_document_chain_3_llm1)

    eval_response_llm1_1 = eval_retrival_chain_1_llm1.invoke({"input":f"Evaluate these questions: {questions_llm1}", "questions": questions_llm1})
    eval_response_llm1_2 = eval_retrival_chain_2_llm1.invoke({"input":f"Evaluate these questions: {questions_llm2}", "questions": questions_llm2})
    eval_response_llm1_3 = eval_retrival_chain_3_llm1.invoke({"input":f"Evaluate these questions: {questions_llm3}", "questions": questions_llm3})

    answers_llm1 = [eval_response_llm1_1, eval_response_llm1_2, eval_response_llm1_3]
    print("|||||"*40)
    print("\n EVALUATION DONE BY THE MODEL DEEPSEEK....\n")

    for ans in answers_llm1:
        print(clean_evaluation_response(ans))
        print("--" * 40)

#-----------------------------------------------------------------------------------------------------------------------

    eval_document_chain_1_llm2 = create_stuff_documents_chain(llm2, ques_prompt_llm1)
    eval_document_chain_2_llm2 = create_stuff_documents_chain(llm2, ques_prompt_llm2)
    eval_document_chain_3_llm2 = create_stuff_documents_chain(llm2, ques_prompt_llm3)

    eval_retrival_chain_1_llm2 = create_retrieval_chain(retriver, eval_document_chain_1_llm2)
    eval_retrival_chain_2_llm2 = create_retrieval_chain(retriver, eval_document_chain_2_llm2)
    eval_retrival_chain_3_llm2 = create_retrieval_chain(retriver, eval_document_chain_3_llm2)

    eval_response_llm2_1 = eval_retrival_chain_1_llm2.invoke(
        {"input": f"Evaluate these questions: {questions_llm1}", "questions": questions_llm1})
    eval_response_llm2_2 = eval_retrival_chain_2_llm2.invoke(
        {"input": f"Evaluate these questions: {questions_llm2}", "questions": questions_llm2})
    eval_response_llm2_3 = eval_retrival_chain_3_llm2.invoke(
        {"input": f"Evaluate these questions: {questions_llm3}", "questions": questions_llm3})

    answers_llm2 = [eval_response_llm2_1, eval_response_llm2_2, eval_response_llm2_3]
    print("|||||" * 40)
    print("\n EVALUATION DONE BY THE MODEL GOOGLW GEMMA 2....\n")

    for ans in answers_llm2:
        print(clean_evaluation_response(ans))
        print("--" * 40)

#-----------------------------------------------------------------------------------------------------------------------

    eval_document_chain_1_llm3 = create_stuff_documents_chain(llm3,ques_prompt_llm1)
    eval_document_chain_2_llm3 = create_stuff_documents_chain(llm3,ques_prompt_llm2)
    eval_document_chain_3_llm3 = create_stuff_documents_chain(llm3,ques_prompt_llm3)

    eval_retrival_chain_1_llm3 = create_retrieval_chain(retriver,eval_document_chain_1_llm3)
    eval_retrival_chain_2_llm3 = create_retrieval_chain(retriver,eval_document_chain_2_llm3)
    eval_retrival_chain_3_llm3 = create_retrieval_chain(retriver,eval_document_chain_3_llm3)

    eval_response_llm3_1 = eval_retrival_chain_1_llm3.invoke({"input":f"Evaluate these questions: {questions_llm1}", "questions": questions_llm1})
    eval_response_llm3_2 = eval_retrival_chain_2_llm3.invoke({"input":f"Evaluate these questions: {questions_llm2}", "questions": questions_llm2})
    eval_response_llm3_3 = eval_retrival_chain_3_llm3.invoke({"input":f"Evaluate these questions: {questions_llm3}", "questions": questions_llm3})

    answers_llm3 = [eval_response_llm3_1, eval_response_llm3_2, eval_response_llm3_3]
    print("|||||"*40)
    print("\n EVALUATION DONE BY THE MODEL OPENAI\OSS ....\n")

    for ans in answers_llm3:
        print(clean_evaluation_response(ans))
        print("--" * 40)



    # print("--" * 40)
    # print("\n EVALUATION DONE BY THE MODEL GOOGLE GEMMA 2....\n")
    # print(eval_response_llm2['answer'])
    #
    # print("--" * 40)
    # print("\n EVALUATION DONE BY THE MODEL OPENAI-OSS....\n")
    # print(eval_response_llm3['answer'])











