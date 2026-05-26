from __future__ import annotations

import pytest

from pipeline.state import InvalidTransition, PipelineStage, PipelineStateMachine


def test_valid_full_sequence():
    sm = PipelineStateMachine()
    stages = [
        PipelineStage.INPUTS_LOADED,
        PipelineStage.DOCUMENTS_CHUNKED,
        PipelineStage.INDEX_BUILT,
        PipelineStage.RETRIEVAL_COMPLETE,
        PipelineStage.DRAFT_ANSWERS_GENERATED,
        PipelineStage.HUMAN_REVIEW_COMPLETE,
        PipelineStage.ANSWERS_AUDITED,
        PipelineStage.FINAL_REPORT_GENERATED,
        PipelineStage.VALIDATION_COMPLETE,
        PipelineStage.RESULTS_FINALISED,
    ]
    for stage in stages:
        sm.advance(stage)
    assert sm.current == PipelineStage.RESULTS_FINALISED


def test_skip_raises():
    sm = PipelineStateMachine()
    with pytest.raises(InvalidTransition):
        sm.advance(PipelineStage.DOCUMENTS_CHUNKED)


def test_backward_raises():
    sm = PipelineStateMachine()
    sm.advance(PipelineStage.INPUTS_LOADED)
    sm.advance(PipelineStage.DOCUMENTS_CHUNKED)
    with pytest.raises(InvalidTransition):
        sm.advance(PipelineStage.INPUTS_LOADED)


def test_current_property():
    sm = PipelineStateMachine()
    assert sm.current == PipelineStage.INIT
    sm.advance(PipelineStage.INPUTS_LOADED)
    assert sm.current == PipelineStage.INPUTS_LOADED
