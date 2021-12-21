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
    def __init__(self, p1_x, p1_y, p2_x, p2_y, t_e=0.0):
        self.p1_x = p1_x
        self.p1_y = p1_y
        self.p2_x = p2_x
        self.p2_y = p2_y
        self.total_error = t_e
        self.alpha = 0.0
        self.positive_points = []
        self.negative_points = []


def get_rule_prediction(a, b, c, x0, y0, check):
    if b > 0:
        a = -a
        b = -b
        c = -c
    temp = a * x0 + b * y0 + c
    # flag indicates the direction of the line
    if check == 1:
        if temp <= 0:
            return True
        else:
            return False
       # return temp <= 0
    else: # check == -1
        if temp >= 0:
            return True
        else:
            return False
     #   return temp >= 0


def line_from_points(p1, p2):
    x_diff = p1[0] - p2[0]
    y_diff = p2[1] - p1[1]
    z = -(y_diff * (p1[0]) + x_diff * (p1[1]))
    return y_diff, x_diff, z


def is_positive(point, rule):
    if point in rule.positive_points:
        return True
    return False


def adaboost(rules_set, points_set):
    max_error = sys.float_info.max
    best_rule = Rule(0, 0, 0, 0, max_error)
    for i in range(len(rules_set)):  # loop each rule
        rules_set[i].total_error = 0.0
        for point in points_set:  # loop each point for each rule
            # if rule predicts + but point actually - or rule predicts - but point actually +
            if (is_positive(point, rules_set[i]) and point.tag == -1) or \
                    (is_positive(point, rules_set[i]) is False and point.tag == 1):
                rules_set[i].total_error += point.weight
            """
            if (point in rules_set[ij].positive_points and point.tag == -1) or \
                    (point not in rules_set[ij].positive_points and point.tag == 1):
                rules_set[ij].total_error += point.weight
            """
        # compute rule`s alpha
        #rules_set[ij].alpha = 0.5 * np.log((1 - rules_set[ij].total_error) / rules_set[ij].total_error)
        # keep rule with lowest weighted error
        numerator = 1 - rules_set[i].total_error
        denominator = rules_set[i].total_error
        alpha_t = 0.5 * np.log(numerator / denominator)
        rules_set[i].alpha = alpha_t

        if rules_set[i].total_error < best_rule.total_error:
            best_rule = rules_set[i]
    return best_rule


def update_weights(best_rule, points):
    weights_sum = 0
    for i in range(len(points)):
        if points[i].x == best_rule.p1_x and points[i].y == best_rule.p1_y or \
                points[i].x == best_rule.p2_x and points[i].y == best_rule.p2_y:
            continue

        success_tag_weight = points[i].weight * np.math.e ** best_rule.alpha
        unsuccess_tag_weight = points[i].weight * np.math.e ** (-1 * best_rule.alpha)

        # two cases of misclassification
        '''if (points[i] in best_rule.positive_points and points[i].tag == -1) or \
                (points[i] not in best_rule.positive_points and points[i].tag == 1):'''
        if (is_positive(points[i], best_rule) and points[i].tag == -1) or \
                (is_positive(points[i], best_rule) is False and points[i].tag == 1):
            points[i].weight = success_tag_weight
            #points[i].weight = points[i].weight * np.exp(best_rule.alpha)
        else:
            points[i].weight = unsuccess_tag_weight
            #points[i].weight = points[i].weight * np.exp(-1 * best_rule.alpha)

        weights_sum += points[i].weight
    for i in range(len(points)):
        points[i].weight = points[i].weight / weights_sum