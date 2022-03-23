from abc import ABC, abstractmethod


class Page(ABC):
    @property
    def name(self) -> str:
        return self._get_name()

    @abstractmethod
    def _get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()
