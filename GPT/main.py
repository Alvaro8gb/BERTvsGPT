import os
import openai
import time
import logging
import traceback
from typing import List
from dotenv import load_dotenv

from GPT.costs import GPT3

from GPT.models import Dialog, Sample, Message, ResultOpenAI
from GPT.costs import ModeloGPT


MAX_TOKENS = 500  # Should be adjust depende on the query
TEMPERATURE = 0.0  # Low temperature to be the more precciss as posible

PARAMS = {'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS}


def generate_messages_NER(behave: Message, input_task: Message, few_shot_samples: List[Sample] = []) -> Dialog:

    if behave["role"] != "system":
        raise Exception("Bad role, should be system")

    if input_task["role"] != "user":
        raise Exception("Bad role, should be user")

    dialog = [behave]

    for sample in few_shot_samples:
        dialog.append(sample.user)
        dialog.append(sample.agent)

    dialog.append(input_task)

    return dialog


class LargeLanguageModel():

    def __init__(self, model: ModeloGPT, **params):
        self.params = params
        self.model = model

        load_dotenv()

        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_type = 'azure'
        openai.api_version = "2023-05-15"

    def query_openai(self, dialog: Dialog):

        response = openai.ChatCompletion.create(
            engine=self.model.name,
            messages=dialog,
            temperature=self.params['temperature'],
            max_tokens=self.params['max_tokens']
        )

        response_message = response.choices[0]["message"]

        logging.info(response)

        prompt_tokens = response["usage"]["prompt_tokens"]
        promt_price = self.model.price_promt * prompt_tokens / 1000
        completion_price = self.model.price_completion * \
            response["usage"]["completion_tokens"] / 1000
        total_price = promt_price+completion_price
        total_tokens = response["usage"]["total_tokens"]
        logging.info("Coste " + str(total_price) + "$")
        logging.info("Numero de tokens " + str(total_tokens))
        

        answer = response_message["content"]
        return ResultOpenAI(price_total=total_price, price_input=promt_price, n_tokens_total=total_tokens, n_tokens_input=prompt_tokens, answer=answer)


def setup_logging():
    # Configure the log format
    logging.basicConfig(
        # Adjust as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='app.log',  # Log file name
        filemode='a',  
        encoding='utf-8'
    )


def log_ner(model: ModeloGPT, behave: Message, input_task: Message, few_shot_samples: List[Sample] = []) -> ResultOpenAI:

    setup_logging()

    try:

        msgs = generate_messages_NER(behave, input_task, few_shot_samples)
        result = model.query_openai(msgs)
        logging.info("Answer: " + str(result.model_dump()))
        time.sleep(0.5)

        return result

    except Exception as e:
        print("See log")
        logging.error(str(e))
        logging.error(traceback.format_exc())
        print("Sleeping")
        time.sleep(10)
        return False


def sample():
    lm = LargeLanguageModel(GPT3, **PARAMS)

    behave = Message(role="system", content="""
                    You are an assistant and an excellent linguist designed to extract the negation entities from the given Text. 
                    Your task is doing NER (Named Entity Recognition) to exctract negated entities. 
                    Extracts all negated entities from the text with and only the following format: @@text negated can be more than one token##.
                    Return me the input text with all negated entities wrapped in the specified format @@ent##. 
                    Respond only with the solution.""")

    input_task = Message(
        role="user", content="""el paciente no tiene cancer de mama y tampoco muestra afeccion cercana""")

    result = log_ner(lm, behave, input_task)

    if result:
        print(result.model_dump())

    else:
        print("Something bad happen")


if __name__ == "__main__":
    sample()