import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.helpers.markov_handler import MarkovHandler
from src.case import Case


class TestMarkovHandler(unittest.TestCase):
    def setUp(self):
        # Initialize parameters for MarkovHandler
        self.nb_tokens = 10
        self.nb_time_steps = 10
        self.bs = 10

    def create_markov_handlers(self, transition_case):
        # Create two instances of MarkovHandler
        markov_handler_false = MarkovHandler(
            self.nb_tokens,
            transition_case,
            self.nb_time_steps,
            store_transition_matrices=False,
        )
        markov_handler_true = MarkovHandler(
            self.nb_tokens,
            transition_case,
            self.nb_time_steps,
            store_transition_matrices=True,
        )
        return markov_handler_false, markov_handler_true

    def generate_t_id(self):
        return torch.randint(0, self.nb_time_steps, (self.bs,))

    def generate_x_id(self):
        return torch.randint(0, self.nb_tokens, (self.bs, 2))

    def run_test(self, function_name, *args):
        # Run the test for both transition cases
        for transition_case in [Case.uniform, Case.absorbing]:
            with self.subTest(transition_case=transition_case):
                (
                    markov_handler_false,
                    markov_handler_true,
                ) = self.create_markov_handlers(transition_case)

                # Call the function for both instances
                result_false = getattr(markov_handler_false, function_name)(
                    *args
                )
                result_true = getattr(markov_handler_true, function_name)(
                    *args
                )
                # Check that both instances return the same result
                torch.testing.assert_close(result_false, result_true)

    def test_extract_rows_Qt(self):
        # Define test inputs
        t_id = self.generate_t_id()
        x_id = self.generate_x_id()
        self.run_test("extract_rows_Qt", t_id, x_id)

    def test_extract_cols_Qt_bar(self):
        # Define test inputs
        t_id = self.generate_t_id()
        x_id = self.generate_x_id()
        self.run_test("extract_cols_Qt_bar", t_id, x_id)

    def test_Qt_x(self):
        # Define test inputs
        t_id = self.generate_t_id()
        x = torch.rand((self.bs, 2, self.nb_tokens))
        self.run_test("Qt_x", t_id, x)


if __name__ == "__main__":
    # test_markov = TestMarkovHandler()
    # test_markov.setUp()
    # test_markov.test_Qt_x()
    unittest.main()
