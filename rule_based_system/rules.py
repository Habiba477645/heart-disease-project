import collections
import collections.abc

collections.Mapping = collections.abc.Mapping
from experta import *


class HeartDiseaseRisk(KnowledgeEngine):
   

    def __init__(self):
        super().__init__()
        # Track how many rules fired per risk category
        self.counts = {"High": 0, "Medium": 0, "Low": 0}
        # Store descriptions of all fired rules
        self.fired_rules = []

    # High Risk Rules

    @Rule(Fact(age=P(lambda x: x > 0.6), chol=P(lambda x: x > 0.6)))
    def high_risk_age_chol(self):
        # Older age combined with high cholesterol increases risk significantly
        print("High Risk: Age and Cholesterol are high")
        self.counts["High"] += 1

    @Rule(Fact(trestbps=P(lambda x: x > 0.6), oldpeak=P(lambda x: x > 0.5)))
    def high_risk_bp_oldpeak(self):
        # High blood pressure with elevated ST depression indicates cardiac stress
        print("High Risk: Blood pressure and oldpeak are high")
        self.counts["High"] += 1

    @Rule(Fact(exang_1=True, oldpeak=P(lambda x: x > 0.5)))
    def high_risk_exercise(self):
        # Exercise-induced angina with ST depression is a strong heart disease indicator
        print("High Risk: Exercise angina detected")
        self.counts["High"] += 1

    #  Medium Risk Rules

    @Rule(Fact(thalach=P(lambda x: x < 0.4)))
    def medium_risk_low_hr(self):
        # Low maximum heart rate may indicate reduced cardiac capacity
        print("Medium Risk: Low maximum heart rate")
        self.counts["Medium"] += 1

    @Rule(Fact(chol=P(lambda x: x > 0.5)))
    def medium_risk_chol(self):
        # Moderately elevated cholesterol is a known risk factor
        print("Medium Risk: Cholesterol is moderate")
        self.counts["Medium"] += 1

    @Rule(Fact(age=P(lambda x: x > 0.5)))
    def medium_risk_age(self):
        # Middle-to-older age groups carry a moderate baseline risk
        print("Medium Risk: Age is moderate")
        self.counts["Medium"] += 1

    #  Low Risk Rules

    @Rule(Fact(trestbps=P(lambda x: x < 0.4), chol=P(lambda x: x < 0.4)))
    def low_risk_bp_chol(self):
        # Normal blood pressure and cholesterol levels suggest low risk
        print("Low Risk: Normal BP and Cholesterol")
        self.counts["Low"] += 1

    @Rule(Fact(thalach=P(lambda x: x > 0.6)))
    def low_risk_hr(self):
        # A high maximum heart rate indicates good cardiovascular fitness
        print("Low Risk: Good heart rate")
        self.counts["Low"] += 1

    @Rule(Fact(oldpeak=P(lambda x: x < 0.3)))
    def low_risk_oldpeak(self):
        # Low ST depression suggests the heart responds well under stress
        print("Low Risk: Normal oldpeak")
        self.counts["Low"] += 1

    @Rule(Fact(exang_1=False))
    def low_risk_exercise(self):
        # Absence of exercise-induced angina is a positive sign
        print("Low Risk: No exercise angina")
        self.counts["Low"] += 1