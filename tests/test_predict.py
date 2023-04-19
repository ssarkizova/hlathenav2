import pandas as pd
import unittest
from unittest.mock import MagicMock
from typing import List
from hlathena import predict

class TestPlotting(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test_predict_no_peptides(self):
        self.assertRaises(IndexError, predict.predict, "path", [])