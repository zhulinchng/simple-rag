from enum import Enum


class PipelineStage(Enum):
    INIT = "INIT"
    INPUTS_LOADED = "INPUTS_LOADED"
    DOCUMENTS_CHUNKED = "DOCUMENTS_CHUNKED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    DRAFT_ANSWERS_GENERATED = "DRAFT_ANSWERS_GENERATED"
    HUMAN_REVIEW_COMPLETE = "HUMAN_REVIEW_COMPLETE"
    ANSWERS_AUDITED = "ANSWERS_AUDITED"
    FINAL_REPORT_GENERATED = "FINAL_REPORT_GENERATED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


_STAGE_ORDER = list(PipelineStage)


class InvalidTransition(RuntimeError):
    pass


class PipelineStateMachine:
    def __init__(self) -> None:
        self._current = PipelineStage.INIT
        print(f"[STATE] {self._current.value}")

    @property
    def current(self) -> PipelineStage:
        return self._current

    def advance(self, expected_next: PipelineStage) -> None:
        current_idx = _STAGE_ORDER.index(self._current)
        next_idx = _STAGE_ORDER.index(expected_next)
        if next_idx != current_idx + 1:
            raise InvalidTransition(
                f"Cannot advance from {self._current.value} to {expected_next.value}; "
                f"expected {_STAGE_ORDER[current_idx + 1].value}"
            )
        self._current = expected_next
        print(f"[STATE] {self._current.value}")
