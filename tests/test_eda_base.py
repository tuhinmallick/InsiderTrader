"""
Tests eda_base.py.
"""
import unittest

import numpy as np
import pandas as pd

from src.insider_eda.eda_base import Exploratory_data_analysis


class TestExploratoryDataAnalysis(unittest.TestCase):

    def test_init(self):
        # test for correct initialization
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        eda = Exploratory_data_analysis(df)
        self.assertIsInstance(eda, Exploratory_data_analysis)
        self.assertEqual(eda.target_name, False)
        self.assertIsInstance(eda.df, pd.DataFrame)
        self.assertTrue(eda.df.equals(df))

        # test for correct initialization with time series data
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6]
            },
            index=pd.date_range(start="2022-01-01", end="2022-01-03"),
        )
        eda = Exploratory_data_analysis(df,
                                        target_name="col1",
                                        time_series=True)
        self.assertIsInstance(eda, Exploratory_data_analysis)
        self.assertEqual(eda.target_name, "col1")
        self.assertIsInstance(eda.df, pd.DataFrame)
        self.assertTrue(eda.df.equals(df))
        self.assertIsInstance(eda.x_date, pd.DatetimeIndex)
        self.assertTrue(eda.y_target.equals(df["col1"]))

        # test for ValueError when index is not datetime
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        },
                          index=[1, 2, 3])
        with self.assertRaises(ValueError):
            eda = Exploratory_data_analysis(df,
                                            target_name="col1",
                                            time_series=True)


if __name__ == "__main__":
    unittest.main()
