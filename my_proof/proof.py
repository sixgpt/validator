import logging
import os
from typing import Dict, Any, List

import random 

from my_proof.extract import extract_data, validate_json_structure
from my_proof.models.proof_response import ProofResponse
from my_proof.wikipedia.verify_content import WikipediaSummarization
from my_proof.sixgpt import evaluate_question, evaluate_answer, get_uniqueness_score

MIN_NUMBER_OF_EXAMPLES = 5
EXAMPLES_PER_API_CALL = 50
MAX_NUMBER_OF_EXAMPLES = 150

def choose_random_example(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    number_of_examples_to_choose = max(len(examples) // EXAMPLES_PER_API_CALL, 1)
    return random.sample(examples, number_of_examples_to_choose)

class Proof:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proof_response = ProofResponse(dlp_id=config['dlp_id'])
        self.avg_question_rating = 0
        self.avg_answer_rating = 0

    def set_nil(self):
        self.proof_response.uniqueness = 0
        self.proof_response.quality = 0
        self.proof_response.authenticity = 0
        self.proof_response.score = 0
        self.proof_response.valid = False

    def proof(self, input_file: str):
        extracted_data = extract_data(input_file)
        if not validate_json_structure(extracted_data):
            logging.info(f"Invalid Data Format")
            self.set_nil()
            return 0, 0
        logging.info(f"Extracted {len(extracted_data)} examples. {extracted_data}")
        number_examples = len(extracted_data)
        if number_examples < MIN_NUMBER_OF_EXAMPLES:
            logging.info(f"Not enough examples")
            self.set_nil()
            return number_examples, 0

        if number_examples > MAX_NUMBER_OF_EXAMPLES:
            logging.info(f"Too many examples")
            self.set_nil()
            return number_examples, 0

        examples = choose_random_example(extracted_data)
        number_sampled = len(examples)
        wikipedia_summarization = WikipediaSummarization()
        for example in examples:
            try:
                content = wikipedia_summarization.get_wikipedia_article_content(example['context']['title'])
                # Start Generation Here
                set_content = set(content.split())
                set_example = set(example['context']['content'].split())
                similarity = len(set_content.intersection(set_example)) / len(set_content.union(set_example)) if set_content.union(set_example) else 0
                logging.info(f"Similarity: {similarity}")
                if (similarity < 0.5):
                    print(f"Similarity too low: {similarity}")
                    self.set_nil()
                    return number_examples, number_sampled

                self.proof_response.authenticity += similarity / number_sampled

                question_rating = evaluate_question(example['input'], content)
                answer_rating = evaluate_answer(example['input'], example['output'], content)
                logging.info(f"Question Rating: {question_rating}, Answer Rating: {answer_rating}")

                self.proof_response.quality += (question_rating * answer_rating) / number_sampled
                self.avg_answer_rating += answer_rating / number_sampled
                self.avg_question_rating += question_rating / number_sampled
            except Exception as e:
                logging.error(f"Error processing example: {e}")
                self.set_nil()
                return number_examples, number_sampled
        
        self.proof_response.uniqueness = get_uniqueness_score([example['context']['title'] for example in examples])
        logging.info(f"Uniqueness: {self.proof_response.uniqueness}")
        return number_examples, number_sampled

    def generate(self) -> ProofResponse:
        """Generate proofs for all input files."""
        logging.info("Starting proof generation")

        # Iterate through files and calculate data validity

        input_filename = os.listdir(self.config['input_dir'])[0]
        number_examples, number_sampled = self.proof(os.path.join(self.config['input_dir'], input_filename))

        # Calculate proof-of-contribution scores: https://docs.vana.org/vana/core-concepts/key-elements/proof-of-contribution/example-implementation
        self.proof_response.ownership = 0.0 
        # Calculate overall score and validity
        total_score = (0.8 * self.proof_response.quality + 0.2 * self.proof_response.uniqueness) * self.proof_response.authenticity
        self.proof_response.score = min(total_score * number_examples / 10_000, 1)
        self.proof_response.valid = total_score > 0

        # Additional (public) properties to include in the proof about the data
        self.proof_response.attributes = {
            'total_score': total_score,
            'number_examples': number_examples,
            'number_sampled': number_sampled,
            'avg_question_rating': self.avg_question_rating,
            'avg_answer_rating': self.avg_answer_rating,
        }

        # Additional metadata about the proof, written onchain
        self.proof_response.metadata = {
            'dlp_id': self.config['dlp_id'],
        }

        return self.proof_response

