import torch
from typing import Dict


def extract_buildings(x: torch.Tensor):
    """ Returns a mask of the buildings in x """
    buildings = x.clone().detach()
    buildings[x > 0] = 1
    return buildings


def compute_tp_fn_fp(pred: torch.Tensor, targ: torch.Tensor, c: int) -> Dict[str, torch.Tensor]:
    """
    Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)
    Args:
        pred (torch.Tensor): prediction
        targ (torch.Tensor): target
        c (int): positive class
    """
    TP = torch.logical_and(pred == c, targ == c).sum()
    FN = torch.logical_and(pred != c, targ == c).sum()
    FP = torch.logical_and(pred == c, targ != c).sum()
    return {"TP": TP, "FN": FN, "FP": FP}


class F1Calculator:
    """
    Records the precision and recall when calculating the f1 score.
    Read about the f1 score here: https://en.wikipedia.org/wiki/F1_score
    """

    def __init__(self, TP, FP, FN, name=''):
        """
        Args:
            TP (int): true positives
            FP (int): false positives
            FN (int): false negatives
            name (str): optional name when printing
        """
        self.TP, self.FN, self.FP, self.name = TP, FN, FP, name
        self.P = self.precision()
        self.R = self.recall()
        self.f1 = self.f1()

    def __repr__(self):
        return f'{self.name} | f1: {self.f1:.4f}, precision: {self.P:.4f}, recall: {self.R:.4f}'

    def precision(self):
        """ calculates the precision using the true positives (self.TP) and false positives (self.FP)"""
        assert self.TP >= 0 and self.FP >= 0
        if self.TP == 0:
            return 0
        else:
            return self.TP/(self.TP+self.FP)

    def recall(self):
        """ calculates recall using the true positives (self.TP) and false negatives (self.FN) """
        assert self.TP >= 0 and self.FN >= 0
        if self.TP == 0:
            return 0
        return self.TP/(self.TP+self.FN)

    def f1(self):
        """ calculates the f1 score using precision (self.P) and recall (self.R) """
        assert 0 <= self.P <= 1 and 0 <= self.R <= 1
        if self.P == 0 or self.R == 0:
            return 0
        return (2*self.P*self.R)/(self.P+self.R)


class Eva:
    def __init__(self, loc_pre: torch.Tensor, des_pre: torch.Tensor, loc_true: torch.Tensor, des_true: torch.Tensor):
        """
            loc_predict: (batch_size, 1, 1024, 1024)
            loc_true: (batch_size, 1, 1024, 1024)
            des_predict: (batch_size, 1, 1024, 1024)
            des_true: (batch_size, 1, 1024, 1024)
            "destroyed": 4,
            "major-damage": 3,
            "minor-damage": 2,
            "no-damage": 1,
            "un-classified": 1
        """
        damage_map = {
            1: "no_damage",
            2: "minor_damage",
            3: "major_damage",
            4: "destroyed"
        }
        self.TP = {"loc": 0, "no_damage": 0, "minor_damage": 0,
                   "major_damage": 0, "destroyed": 0}
        self.FN = {"loc": 0, "no_damage": 0, "minor_damage": 0,
                   "major_damage": 0, "destroyed": 0}
        self.FP = {"loc": 0, "no_damage": 0, "minor_damage": 0,
                   "major_damage": 0, "destroyed": 0}
        for i in range(loc_pre.shape[0]):
            loc_pre_binary = extract_buildings(loc_pre[i])
            loc_true_binary = extract_buildings(loc_true[i])
            des_true_binary = extract_buildings(des_true[i])
            # only give credit to damages where buildings are predicted
            des_pre[i] = des_pre[i] * loc_pre_binary
            
            # only score damage where there exist buildings in target damage
            des_pre[i] = des_pre[i] * des_true_binary
            des_true[i] = des_true[i] * des_true_binary

            temp = compute_tp_fn_fp(loc_pre_binary, loc_true_binary, 1)
            self.TP["loc"] += temp["TP"]
            self.FN["loc"] += temp["FN"]
            self.FP["loc"] += temp["FP"]

            for j in range(1, 5):
                temp = compute_tp_fn_fp(des_pre[i], des_true[i], j)
                self.TP[damage_map[j]] += temp["TP"]
                self.FN[damage_map[j]] += temp["FN"]
                self.FP[damage_map[j]] += temp["FP"]

    def loc_f1(self):
        return F1Calculator(self.TP["loc"], self.FP["loc"], self.FN["loc"], name="loc_f1").f1

    def damage_f1(self):
        no_damage_f1 = F1Calculator(
            self.TP["no_damage"], self.FP["no_damage"], self.FN["no_damage"], name="no_damage_f1").f1
        minor_damage_f1 = F1Calculator(
            self.TP["minor_damage"], self.FP["minor_damage"], self.FN["minor_damage"], name="minor_damage_f1").f1
        major_damage_f1 = F1Calculator(
            self.TP["major_damage"], self.FP["major_damage"], self.FN["major_damage"], name="major_damage_f1").f1
        destroyed_f1 = F1Calculator(
            self.TP["destroyed"], self.FP["destroyed"], self.FN["destroyed"], name="destroyed_f1").f1

        def harmonic_mean(xs): return len(xs) / sum((x+1e-6)**-1 for x in xs)
        return harmonic_mean([no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1])

    def __iadd__(self, other: "Eva"):
        for k in self.FP.keys():
            self.FP[k] += other.FP[k]
            self.FN[k] += other.FN[k]
            self.TP[k] += other.TP[k]
        return self

class Eva_loc:
    def __init__(self, loc_pre: torch.Tensor, loc_true: torch.Tensor):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        for i in range(loc_pre.shape[0]):
            loc_pre_binary = extract_buildings(loc_pre[i])
            loc_true_binary = extract_buildings(loc_true[i])
            temp = compute_tp_fn_fp(loc_pre_binary, loc_true_binary, 1)
            self.TP += temp["TP"]
            self.FN += temp["FN"]
            self.FP += temp["FP"]
    def f1(self):
        return F1Calculator(self.TP, self.FP, self.FN, name="loc_f1").f1
    def __iadd__(self, other: "Eva_loc"):
        self.FP += other.FP
        self.FN += other.FN
        self.TP += other.TP
        return self