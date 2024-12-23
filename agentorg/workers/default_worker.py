import logging
import os

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agentorg.workers.worker import BaseWorker, register_worker, WORKER_REGISTRY
from agentorg.workers.prompts import choose_worker_prompt
from agentorg.utils.utils import chunk_string
from agentorg.utils.graph_state import MessageState
from agentorg.utils.model_config import MODEL


logger = logging.getLogger(__name__)


@register_worker
class DefaultWorker(BaseWorker):

    description = "Default worker decided by chat records if there is no specific worker for the user's query"

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
        self.base_choice = "CheckerWorker"
        available_workers = os.getenv("AVAILABLE_WORKERS", "").split(",")
        self.available_workers = {name: WORKER_REGISTRY[name].description for name in available_workers if name != "DefaultWorker"}

    def _choose_worker(self, state: MessageState, limit=2):
        user_message = state['user_message']
        task = state["orchestrator_message"].attribute.get("task", "")
        workers_info = "\n".join([f"{name}: {description}" for name, description in self.available_workers.items()])
        workers_name = ", ".join(self.available_workers.keys())

        prompt = PromptTemplate.from_template(choose_worker_prompt)
        input_prompt = prompt.invoke({"message": user_message.message, "formatted_chat": user_message.history, "task": task, "workers_info": workers_info, "workers_name": workers_name})
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        while limit > 0:
            answer = final_chain.invoke(chunked_prompt)
            for worker_name in self.available_workers.keys():
                if worker_name in answer:
                    logger.info(f"Chosen worker for the default worker: {worker_name}")
                    return worker_name
            limit -= 1
        logger.info(f"Base worker chosen for the default worker: {self.base_choice}")
        return self.base_choice
    
    def execute(self, msg_state: MessageState):
        chose_worker = self._choose_worker(msg_state)
        worker = WORKER_REGISTRY[chose_worker]()
        result = worker.execute(msg_state)
        return result
