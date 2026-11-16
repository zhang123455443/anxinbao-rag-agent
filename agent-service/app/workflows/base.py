from abc import ABC, abstractmethod

from langgraph.graph import StateGraph


class Workflow(ABC):

    @abstractmethod
    def workflow_name(self) -> str:
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass