"""combination of message worker, rag message worker, and default worker"""
import logging
import os

from langgraph.graph import StateGraph, START
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agentorg.workers.worker import BaseWorker, register_worker, WORKER_REGISTRY
from agentorg.workers.prompts import checker_generator_prompt, choose_worker_prompt
from agentorg.utils.utils import chunk_string
from agentorg.utils.graph_state import MessageState
from agentorg.utils.model_config import MODEL


logger = logging.getLogger(__name__)


@register_worker
class CheckerWorker(BaseWorker):

    description = "A worker that modifies the prompt to include previous mistakes and then picks the best worker"

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(model=MODEL["model_type_or_path"], timeout=30000)
        self.base_choice = "MessageWorker"
        available_workers = os.getenv("AVAILABLE_WORKERS", "").split(",")
        self.available_workers = {name: WORKER_REGISTRY[name].description for name in available_workers if name != "DefaultWorker" and name != "CheckerWorker"}
        self.previous_checks = ""

    def checker(self, state: MessageState) -> MessageState:
        # get the input message
        user_message = state['user_message']
        orchestrator_message = state['orchestrator_message']

        # get the orchestrator message content
        orch_msg_content = orchestrator_message.message
        orch_msg_attr = orchestrator_message.attribute
        direct_response = orch_msg_attr.get('direct_response', False)
        if direct_response:
            return orch_msg_content

        # generate answer based on the retrieved texts
        prompt = PromptTemplate.from_template(checker_generator_prompt)
        input_prompt = prompt.invoke({"sys_instruct": state["sys_instruct"], "message": orch_msg_content, "formatted_chat": user_message.history})
        chunked_prompt = chunk_string(input_prompt.text, tokenizer=MODEL["tokenizer"], max_length=MODEL["context"])
        final_chain = self.llm | StrOutputParser()
        logger.info(f"Prompt: {input_prompt.text}")
        answer = final_chain.invoke(chunked_prompt) + self.previous_checks
        self.previous_checks = answer
        state["message_flow"] = answer
        return state
    
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

    def _create_action_graph(self, msg_state):
        workflow = StateGraph(MessageState)
        # Add nodes for each worker
        chose_worker = self._choose_worker(msg_state)
        worker = WORKER_REGISTRY[chose_worker]()
        workflow.add_node("checker", self.checker)
        workflow.add_node("chosen_worker", worker.execute)
        # Add edges
        workflow.add_edge(START, "checker")
        workflow.add_edge("checker", "chosen_worker")
        return workflow

    def execute(self, msg_state: MessageState):
        graph = self._create_action_graph(msg_state).compile()
        result = graph.invoke(msg_state)
        return result