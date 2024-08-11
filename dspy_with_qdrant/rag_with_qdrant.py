import ollama
import dspy
from qdrant_client import QdrantClient
import json

# Read parameters file
parameters_file = "params.json"
with open(parameters_file, 'r') as fh:
    params = json.loads(fh.read())



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

    # Connect to the Qdrant vector db
    client = QdrantClient("localhost", port=6333)

    # Fetch similar records from vector DB
    hits = client.search(
        collection_name=params["collection_name"],
        query_vector=query_vector,
        limit=num_passages
    )

    # Collect context data from the collected records
    context = []

    for x in hits:
        context.append(x.payload['text'])

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

    def __init__(self, num_passages: int=5):
        super().__init__()

        self.num_passages = num_passages
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context = get_context(self.num_passages, question)
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)