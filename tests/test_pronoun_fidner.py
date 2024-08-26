import unittest
import pandas as pd
import os
from io import StringIO

from GUI_V2 import run_script

class TestPronounFinder(unittest.TestCase):
    def setUp(self):
        # Create a temporary input CSV file
        self.input_csv = "test_input.csv"
        self.output_csv = "test_output.csv"
        with open(self.input_csv, "w") as f:
            f.write("This is a test sentence. She loves programming.")

    def tearDown(self):
        # Remove the temporary files
        if os.path.exists(self.input_csv):
            os.remove(self.input_csv)
        if os.path.exists(self.output_csv):
            os.remove(self.output_csv)

    def test_single_sentence(self):
        # Run the script
        run_script(self.input_csv, self.output_csv)

        # Read the output CSV file
        result_df = pd.read_csv(self.output_csv)

        # Check the output
        self.assertFalse(result_df.empty, "The output CSV should not be empty.")
        self.assertIn("She", result_df["Pronoun"].values, "The pronoun 'She' should be in the output.")
        self.assertIn("a test sentence", result_df["Candidate Antecedent"].values, "The candidate antecedent 'a test sentence' should be in the output.")

if __name__ == "__main__":
    unittest.main()