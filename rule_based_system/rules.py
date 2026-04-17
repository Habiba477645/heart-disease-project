import collections
import collections.abc

collections.Mapping = collections.abc.Mapping

from experta import *


class HeartDiseaseRisk(KnowledgeEngine):
    

    def __init__(self):
        super().__init__()
        
        self.counts = {"High": 0, "Medium": 0, "Low": 0}
        
        self.fired_rules = []

    # High Risk Rules

    @Rule(Fact(age=P(lambda x: x > 0.6), chol=P(lambda x: x > 0.6)))
    def high_risk_age_chol(self):
        self.counts['High'] += 1

    @Rule(Fact(trestbps=P(lambda x: x > 0.6), exang_1=True))
    def high_risk_exercise(self):
        self.counts['High'] += 1

    @Rule(Fact(exang_1=True, oldpeak=P(lambda x: x > 0.5)))
    def high_risk_exercise(self):
       
        print("High Risk: Exercise angina detected")
        self.counts["High"] += 1

    #  Medium Risk Rules

    @Rule(Fact(thalach=P(lambda x: x < 0.4)))
    def medium_risk_low_hr(self):
       
        print("Medium Risk: Low maximum heart rate")
        self.counts["Medium"] += 1

    @Rule(Fact(chol=P(lambda x: x > 0.5)))
    def medium_risk_chol(self):
       
        print("Medium Risk: Cholesterol is moderate")
        self.counts["Medium"] += 1

    @Rule(Fact(age=P(lambda x: x > 0.5)))
    def medium_risk_age(self):
       
        print("Medium Risk: Age is moderate")
        self.counts["Medium"] += 1

    #  Low Risk Rules

    @Rule(Fact(trestbps=P(lambda x: x < 0.4), chol=P(lambda x: x < 0.4)))
    def low_risk_bp_chol(self):
       
        print("Low Risk: Normal BP and Cholesterol")
        self.counts["Low"] += 1

    @Rule(Fact(thalach=P(lambda x: x > 0.6)))
    def low_risk_hr(self):
        self.counts['Low'] += 1

    @Rule(Fact(oldpeak=P(lambda x: x < 0.3)))
    def low_risk_oldpeak(self):
       
        print("Low Risk: Normal oldpeak")
        self.counts["Low"] += 1

    @Rule(Fact(exang_1=False))
    def low_risk_exercise(self):
       
        print("Low Risk: No exercise angina")
        self.counts["Low"] += 1