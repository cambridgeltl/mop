import unittest

from transformers.data.processors.squad import SquadExample
from transformers.pipelines import Pipeline, QuestionAnsweringArgumentHandler

from .test_pipelines_common import CustomInputPipelineCommonMixin


class QAPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "question-answering"
    small_models = [
        "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    ]  # Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"score", "answer", "start", "end"}
        valid_inputs = [
            {"question": "Where was HuggingFace founded ?", "context": "HuggingFace was founded in Paris."},
            {
                "question": "In what field is HuggingFace working ?",
                "context": "HuggingFace is a startup based in New-York founded in Paris which is trying to solve NLP.",
            },
        ]
        invalid_inputs = [
            {"question": "", "context": "This is a test to try empty question edge case"},
            {"question": None, "context": "This is a test to try empty question edge case"},
            {"question": "What is does with empty context ?", "context": ""},
            {"question": "What is does with empty context ?", "context": None},
        ]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, dict)

        for key in output_keys:
            self.assertIn(key, mono_result)

        multi_result = nlp(valid_inputs)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)
        for bad_input in invalid_inputs:
            self.assertRaises(ValueError, nlp, bad_input)
        self.assertRaises(ValueError, nlp, invalid_inputs)

    def test_argument_handler(self):
        qa = QuestionAnsweringArgumentHandler()

        Q = "Where was HuggingFace founded ?"
        C = "HuggingFace was founded in Paris"

        normalized = qa(Q, C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(question=Q, context=C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(question=Q, context=C)
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa({"question": Q, "context": C})
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa([{"question": Q, "context": C}])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa([{"question": Q, "context": C}, {"question": Q, "context": C}])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 2)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(X={"question": Q, "context": C})
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(X=[{"question": Q, "context": C}])
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

        normalized = qa(data={"question": Q, "context": C})
        self.assertEqual(type(normalized), list)
        self.assertEqual(len(normalized), 1)
        self.assertEqual({type(el) for el in normalized}, {SquadExample})

    def test_argument_handler_error_handling(self):
        qa = QuestionAnsweringArgumentHandler()

        Q = "Where was HuggingFace founded ?"
        C = "HuggingFace was founded in Paris"

        with self.assertRaises(KeyError):
            qa({"context": C})
        with self.assertRaises(KeyError):
            qa({"question": Q})
        with self.assertRaises(KeyError):
            qa([{"context": C}])
        with self.assertRaises(ValueError):
            qa(None, C)
        with self.assertRaises(ValueError):
            qa("", C)
        with self.assertRaises(ValueError):
            qa(Q, None)
        with self.assertRaises(ValueError):
            qa(Q, "")

        with self.assertRaises(ValueError):
            qa(question=None, context=C)
        with self.assertRaises(ValueError):
            qa(question="", context=C)
        with self.assertRaises(ValueError):
            qa(question=Q, context=None)
        with self.assertRaises(ValueError):
            qa(question=Q, context="")

        with self.assertRaises(ValueError):
            qa({"question": None, "context": C})
        with self.assertRaises(ValueError):
            qa({"question": "", "context": C})
        with self.assertRaises(ValueError):
            qa({"question": Q, "context": None})
        with self.assertRaises(ValueError):
            qa({"question": Q, "context": ""})

        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": None, "context": C}])
        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": "", "context": C}])

        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": Q, "context": None}])
        with self.assertRaises(ValueError):
            qa([{"question": Q, "context": C}, {"question": Q, "context": ""}])

    def test_argument_handler_error_handling_odd(self):
        qa = QuestionAnsweringArgumentHandler()
        with self.assertRaises(ValueError):
            qa(None)

        with self.assertRaises(ValueError):
            qa(Y=None)

        with self.assertRaises(ValueError):
            qa(1)
