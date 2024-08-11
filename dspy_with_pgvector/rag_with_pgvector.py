import ollama
import dspy
import json
import psycopg2


# Read parameters file
parameters_file = "pg_params.json"
with open(parameters_file, 'r') as fh:
    params = json.loads(fh.read())


def combine_data(data_all):

    for x in data_all:
        print(x[0], x[3], x[2][:100])

    source = ""
    text = ""
    data_combined = []

    for x in data_all:
        if source != x[0]:
            source = x[0]
            data_combined.append(text)
            text = x[2]
        elif source == x[0]:
            text += x[2]

    data_combined.pop(0)
    data_combined.append(text)

    return data_combined



# 1. Function to retrieve context records from vector db
def get_context(num_passages: int, text: str) -> list:
    '''Get context from the vector DB for the provided question.
    
    Parameters
    ----------
    num_passages (int): 
        No. of similar records to retrieve from the vector database
    text (str): 
        Question to retrieve context for 

    Returns
    -------
    context (list): 
        A list of context data
    '''

    # Create embedding for the question
    query_vector = ollama.embeddings(model=params["embedding_model"], prompt=str(text))["embedding"]

    # Connect to the database
    conn = psycopg2.connect(database = "test_db", 
                        user = "postgres", 
                        host= 'localhost',
                        password = "deep",
                        port = 5432)

    # Get similar records from table
    if params['table_name'] == 'college_text':
        with conn.cursor() as cur:
            cur.execute(f"SELECT doc_text FROM college_text ORDER BY '{query_vector}' <=> embedding LIMIT {num_passages};")
            context = cur.fetchall()

    elif params['table_name'] == 'college_data':
        with conn.cursor() as cur:
            cur.execute(f"""SELECT doc_text FROM college_data
                    WHERE doc_name IN (SELECT doc_name FROM college_data ORDER BY 1 - ('{query_vector}' <=> embedding) LIMIT {num_passages});""")
            context = cur.fetchall()

    elif params['table_name'] == 'college_ordered':
        with conn.cursor() as cur:
            cur.execute(f"""SELECT doc_name, doc_order, doc_text, 1 - ('{query_vector}' <=> embedding) AS cosine_distance FROM college_ordered
                    WHERE doc_name IN (SELECT doc_name FROM college_ordered ORDER BY 1 - ('{query_vector}' <=> embedding) LIMIT {num_passages}) ORDER BY doc_name, doc_order;""")
            data_all = cur.fetchall()

        context = combine_data(data_all)

    print(len(context))


    # Close the connection to the database
    conn.close()

    
    return context



# 2. Metric

# Metric Prompt Signature
class TypedEvaluator(dspy.Signature):
    """Evaluate the quality of a systems answer to a question according to a given criterion."""

    criterion = dspy.InputField(desc="The evaluation criterion")
    question = dspy.InputField(desc="The question asked to the system.")
    ground_truth_answer = dspy.InputField(desc="An expert written Ground Truth Answer to the question.")
    predicted_answer = dspy.InputField(desc="The system's answer to the question.")
    rating = dspy.OutputField(desc="A float rating between 1 and 5. IMPORTANT!! ONLY OUTPUT THE RATING")

# Metric Function
def MetricWrapper(sample, pred, trace=None) -> bool:
    alignment_criterion = "How aligned is the predicted_answer with the ground_truth?"

    score = dspy.TypedPredictor(TypedEvaluator)(criterion=alignment_criterion,
                                               question=sample.question,
                                               ground_truth_answer=sample.answer,
                                               predicted_answer=pred.answer).rating
    
    return float(score) >= 4



# 3. RAG

# RAG Prompt Signature
class GenerateAnswer(dspy.Signature):
    """Answer the questions directly using clear and concise language and do not provide extra details. 
    If the answer is not found in the context, state: "For the above information, kindly reach out to academics or student support.
    """

    context = dspy.InputField(desc="Context that contains all the relevant facts needed to answer the question.")
    question = dspy.InputField(desc="The question asked to the system.")
    answer = dspy.OutputField(desc="Answer in detail. IMPORTANT! ONLY OUTPUT THE ANSWER.")

# RAG Signature
class RAG(dspy.Module):
    '''RAG model with Chain of Thought'''

    def __init__(self, num_passages: int=8):
        super().__init__()

        self.num_passages = num_passages
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context = get_context(self.num_passages, question)
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)