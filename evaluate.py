import os
import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from rag.llm import get_llm
from rag.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation(eval_data: list):
    """Run RAGAS evaluation on a provided dataset.
    
    Expected format for eval_data:
    [
        {
            "question": "What is hybrid search?",
            "answer": "Hybrid search combines dense and sparse retrieval.",
            "contexts": ["Hybrid search merges FAISS and BM25...", "Another relevant chunk..."],
            "ground_truth": "Hybrid search is a technique that uses both semantic and keyword search."
        },
        ...
    ]
    """
    logger.info("Initializing evaluation with RAGAS...")
    
    if not eval_data:
        logger.error("No evaluation data provided.")
        return None

    # Load LLM and Embeddings from the existing project setup
    llm = get_llm()
    embeddings = get_embeddings()

    # Create HuggingFace Dataset required by Ragas
    dataset = Dataset.from_list(eval_data)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    logger.info(f"Running evaluation on {len(eval_data)} examples...")
    
    # Run evaluation
    # Note: Depending on the specific version of Ragas, llm and embeddings might need to be wrapped.
    # In newer ragas versions, providing the standard Langchain objects works seamlessly.
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        
        logger.info("Evaluation complete!")
        print("\n--- RAGAS Evaluation Results ---")
        for metric, score in results.items():
            print(f"{metric.replace('_', ' ').title()}: {score:.4f}")
            
        return results
    except Exception as e:
        logger.error(f"Error during RAGAS evaluation: {str(e)}")
        return None

if __name__ == "__main__":
    # Sample usage block for testing the evaluation pipeline independently
    sample_data = [
        {
            "question": "What is the purpose of RRF?",
            "answer": "Reciprocal Rank Fusion merges two ranked result lists.",
            "contexts": ["Reciprocal Rank Fusion (RRF) merges faiss_results and bm25_results by accumulating inverse rank scores."],
            "ground_truth": "RRF combines rankings from multiple retrieval methods into a single fused list."
        }
    ]
    
    print("Starting sample evaluation...")
    run_evaluation(sample_data)
