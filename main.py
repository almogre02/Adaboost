import numpy as np
import time
import random
from adaboost import Rule, Point, adaboost, line_from_points, get_rule_prediction, update_weights, is_positive

if __name__ == '__main__':
    Iterations = 100
    K = 8
    points_list = []
    true_error_test_set = [0.0]*K
    empirical_error_training_set = [0.0]*K
    # read file
    f = open("../../Desktop/rectangle.txt")
    read_lines = f.read().splitlines()

    # split lines into array '1.25 2.50 1' --> [1.25, 2.50, 1]
    for i in range(len(read_lines)):
        points_list.insert(i, read_lines[i].split(' '))

    # create array containing all Points
    points_arr = []
    for i in range(len(points_list)):
        points_arr.insert(i, Point(float(points_list[i][0]), float(points_list[i][1]), int(points_list[i][2])))
    # avoid divide by zero exception
    np.seterr(divide='ignore')

    # create array containing all possible lines (rules)
    rules = []
    for i in range(len(points_list)):
        for j in range(0, len(points_list)):
            if i != j:
                rules.insert(i, Rule(float(points_list[i][0]), float(points_list[i][1]), float(points_list[j][0]), float(points_list[j][1])))
    #  set for every rule the set of points tagged as +
    for rule in rules:  # loop each rule
        a1, b1, c1 = line_from_points([rule.p1_x, rule.p1_y], [rule.p2_x, rule.p2_y])
        for p in points_arr:  # loop each point for each rule
            p1_distance = np.sqrt(rule.p1_x ** 2 + rule.p1_y ** 2)
            p2_distance = np.sqrt(rule.p2_x ** 2 + rule.p2_y ** 2)
            if p1_distance < p2_distance:
                check = 1
            else:
                check = -1
            if get_rule_prediction(a1, b1, c1, p.x, p.y, check):
                rule.positive_points.append(p)
            else:
                rule.negative_points.append(p)
    start_time=time.perf_counter()#

    for y in range(Iterations):
        best_rules = []
        # split data randomly into two halves
        random.shuffle(points_arr)
        split_size = int(len(points_arr) / 2)

        training_set = points_arr[split_size:]
        test_set = points_arr[:split_size]

        # init points weights
        for p in training_set:
            p.weight = 1 / len(training_set)

        # execute 8 iteration of adaboost to find the 8 best rules (chosen by the biggest alpha)
        for i in range(K):
            adaboost_best_rule = adaboost(rules, training_set)
            best_rules.insert(i, [adaboost_best_rule, adaboost_best_rule.alpha])
            update_weights(training_set, adaboost_best_rule)

        H_k = []
        for i in range(0, K):
            rules_iteration = []
            for j in range(0, i+1):
                rules_iteration.append(best_rules[i])
            H_k.insert(i, rules_iteration)
        # build Hk(x) rules
        '''H_k = [[best_rules[0]],
               [best_rules[0], best_rules[1]],
               [best_rules[0], best_rules[1], best_rules[2]],
               [best_rules[0], best_rules[1], best_rules[2], best_rules[3]],
               [best_rules[0], best_rules[1], best_rules[2], best_rules[3], best_rules[4]],
               [best_rules[0], best_rules[1], best_rules[2], best_rules[3], best_rules[4], best_rules[5]],
               [best_rules[0], best_rules[1], best_rules[2], best_rules[3], best_rules[4], best_rules[5], best_rules[6]],
               [best_rules[0], best_rules[1], best_rules[2], best_rules[3], best_rules[4], best_rules[5], best_rules[6], best_rules[7]]]'''

        # get Hk(x) final prediction on every point
        for p in points_arr:
            p.final_prediction = []
            for i in range(len(H_k)):
                final_prediction = 0.0
                for j in range(len(H_k[i])):
                    if is_positive(p, H_k[i][j][0]):
                    #if p in Hk[i][j][0].positive_points:
                        prediction = 1
                    else:
                        prediction = -1
                    final_prediction += H_k[i][j][1] * prediction
                p.final_prediction.insert(i, np.sign(final_prediction))

        # compute empirical error of every Hk(x)
        # empirical error  = num of misclassified points in train set / num of points
        for i in range(K):
            training_counter = 0
            for p2 in training_set:
                if (p2.tag == 1 and p2.final_prediction[i] == -1) or \
                        (p2.tag == -1 and p2.final_prediction[i] == 1):
                    training_counter += 1
            empirical_error_training_set[i] += training_counter / len(training_set)

        # compute true error of every Hk(x)
        # true error  = num of misclassified points in test set / num of points

        for i in range(K):
            test_counter = 0
            for p2 in test_set:
                if (p2.tag == 1 and p2.final_prediction[i] == -1) or \
                        (p2.tag == -1 and p2.final_prediction[i] == 1):  # two cases of misclassification
                    test_counter += 1
            true_error_test_set[i] += test_counter / len(test_set)

    end_time=time.perf_counter()
    print(end_time-start_time)
    for k in range(K):
        print("Training Set: the empirical error of the training set for #{} rules is: {} ".format(k+1, empirical_error_training_set[k] / Iterations))
        print("Test Set: the true error of the test set for #{} rules is: {} ".format(k+1, true_error_test_set[k] / Iterations))
        print("")

       #print("true_error avg for ", ADABOOST_ITERATIONS, " runs: ", true_error_test_set[k] / ADABOOST_ITERATIONS,
        #      ", empirical_error: ", empirical_error_training_set[k] / ADABOOST_ITERATIONS)
