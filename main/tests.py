import os
from pathlib import Path

import joblib
from django.test import TestCase
# Create your tests here.
class maintests(TestCase):
    # knowhow_model = joblib.load(os.path.join(Path(__file__).resolve().parent, f'ai/knowhow_ai19.pkl'))

    model = joblib.load('knowhow_ai.pkl')

    print(model)