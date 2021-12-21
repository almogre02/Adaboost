import sys
import numpy as np


class Point:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.weight = 0.0
        self.tag = t
        self.final_prediction = []


class Rule:
    def __init__(self, x1, y1, x2, y2, t_e=0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.total_error = t_e
        self.alpha = 0.0
        self.positive_points = []
        self.negative_points = []


def get_rule_prediction(a, b, c, x0, y0, flag):
    if b > 0:
        a = -a
        b = -b
        c = -c
    temp = a * x0 + b * y0 + c
    # flag indicates the direction of the line
    if flag:
        if temp <= 0:
            return True
        else:
            return False
       # return temp <= 0
    else:
        if temp >= 0:
            return True
        else:
            return False
     #   return temp >= 0


def line_from_points(p1, q):
    a = q[1] - p1[1]
    b = p1[0] - q[0]
    c = -(a * (p1[0]) + b * (p1[1]))
    return a, b, c


def is_positive(point, rule):
    if point in rule.positive_points:
        return True
    return False


def adaboost(rules_set, points_set):
    max_error = sys.float_info.max
    best_rule = Rule(0, 0, 0, 0, max_error)
    for ij in range(len(rules_set)):  # loop each rule
        rules_set[ij].total_error = 0.0
        for point in points_set:  # loop each point for each rule
            # if rule predicts + but point actually - or rule predicts - but point actually +
            if (is_positive(point, rules_set[ij]) and point.tag == -1) or \
                    (not is_positive(point, rules_set[ij]) and point.tag == 1):
                rules_set[ij].total_error += point.weight
            """
            if (point in rules_set[ij].positive_points and point.tag == -1) or \
                    (point not in rules_set[ij].positive_points and point.tag == 1):
                rules_set[ij].total_error += point.weight
            """
        # compute rule`s alpha
        #rules_set[ij].alpha = 0.5 * np.log((1 - rules_set[ij].total_error) / rules_set[ij].total_error)
        # keep rule with lowest weighted error
        numerator = 1 - rules_set[ij].total_error
        denominator = rules_set[ij].total_error
        alpha_t = 0.5 * np.log(numerator / denominator)
        rules_set[ij].alpha = alpha_t

        if rules_set[ij].total_error < best_rule.total_error:
            best_rule = rules_set[ij]
    return best_rule


def update_weights(best_rule, points):
    weights_sum = 0
    for u in range(len(points)):
        if points[u].x == best_rule.x1 and points[u].y == best_rule.y1 or \
                points[u].x == best_rule.x2 and points[u].y == best_rule.y2:
            continue

        success_tag_weight = points[u].weight * np.math.e ** best_rule.alpha
        unsuccess_tag_weight = points[u].weight * np.math.e ** (-1 * best_rule.alpha)

        # two cases of misclassification
        '''if (points[u] in best_rule.positive_points and points[u].tag == -1) or \
                (points[u] not in best_rule.positive_points and points[u].tag == 1):'''
        if (is_positive(points[u], best_rule) and points[u].tag == -1) or \
                (is_positive(points[u], best_rule) is False and points[u].tag == 1):
            points[u].weight = success_tag_weight
            #points[u].weight = points[u].weight * np.exp(best_rule.alpha)
        else:
            points[u].weight = unsuccess_tag_weight
            #points[u].weight = points[u].weight * np.exp(-1 * best_rule.alpha)

        weights_sum += points[u].weight
    for u in range(len(points)):
        points[u].weight = points[u].weight / weights_sum