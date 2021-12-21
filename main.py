import numpy as np
import time
import random
from adaboost import Rule, Point, adaboost, line_from_points, get_rule_prediction, update_weights, is_positive

if __name__ == '__main__':
    iterations = 100
    K = 8
    #true_error, empirical_error = [0 for _ in range(8)], [0 for _ in range(8)]
    true_error = [0.0]*K
    empirical_error = [0.0]*K
    points_list = []
    rules = []
    points_set = []
    # read file
    f = open("../../Desktop/rectangle.txt")
    read_lines = f.read().splitlines()

    # split lines into array '1.25 2.50 1' --> [1.25, 2.50, 1]

    for i in range(len(read_lines)):
        points_list.insert(i, read_lines[i].split(' '))

    # create array containing all Points

    for i in range(len(points_list)):
        new_point = Point(float(points_list[i][0]), float(points_list[i][1]), int(points_list[i][2]))
        points_set.insert(i, new_point)
    # avoid divide by zero exception
    np.seterr(divide='ignore')

    # create array containing all possible lines (rules)

    for i in range(len(points_list)):
        for j in range(0, len(points_list)):
            if i != j:
                new_rule = Rule(float(points_list[i][0]), float(points_list[i][1]), float(points_list[j][0]), float(points_list[j][1]))
                rules.insert(i, new_rule)
    #  set for every rule the set of points tagged as +
    for r in rules:  # loop each rule
        p1 = [r.p1_x, r.p1_y]
        p2 = [r.p2_x, r.p2_y]
        y_diff, x_diff, z = line_from_points(p1, p2)
        for p in points_set:  # loop each point for each rule
            p1_distance = np.sqrt(r.p1_x ** 2 + r.p1_y ** 2)
            p2_distance = np.sqrt(r.p2_x ** 2 + r.p2_y ** 2)
            if p1_distance < p2_distance:
                check = 1
            else:
                check = -1

            if get_rule_prediction(y_diff, x_diff, z, p.x, p.y, check):
                r.positive_points.append(p)
            else:
                r.negative_points.append(p)
    start_time = time.perf_counter()  #

    for x in range(iterations):

        # split data randomly into two halves
        random.shuffle(points_set)
        arr_size = int(len(points_set) / 2)
        train_set = points_set[arr_size:]
        test_set = points_set[:arr_size]

        # init points weights
        for train_set_point in train_set:
            train_set_point.weight = 1 / len(train_set)

        # execute 8 iteration of adaboost to find the 8 best rules (chosen by the biggest alpha)
        results = []
        for i in range(K):
            result = adaboost(rules, train_set)
            results.insert(i, [result, result.alpha])
            update_weights(result, train_set)

        # build Hk(x) rules
        '''Hk = [[results[0]],
              [results[0], results[1]],
              [results[0], results[1], results[2]],
              [results[0], results[1], results[2], results[3]],
              [results[0], results[1], results[2], results[3], results[4]],
              [results[0], results[1], results[2], results[3], results[4], results[5]],
              [results[0], results[1], results[2], results[3], results[4], results[5], results[6]],
              [results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]]'''
        Hk = []
        for i in range(0, 8):
            rules_iteration = []
            for j in range(0, i + 1):
                rules_iteration.insert(j, results[j])
            Hk.insert(i, rules_iteration)

        # get Hk(x) final prediction on every point
        for p in points_set:
            p.final_prediction = []
            for i in range(len(Hk)):
                final_prediction = 0.0
                for j in range(len(Hk[i])):
                    if is_positive(p, Hk[i][j][0]): #p in Hk[i][j][0].positive_points:
                        prediction = 1
                    else:
                        prediction = -1
                    final_prediction += Hk[i][j][1] * prediction
                p.final_prediction.insert(i, np.sign(final_prediction))

        # compute empirical error of every Hk(x)
        # empirical error  = num of misclassified points in train set / num of points
        for i in range(K):
            counter = 0
            for p2 in train_set:
                if (p2.tag == 1 and p2.final_prediction[i] == -1) or \
                        (p2.tag == -1 and p2.final_prediction[i] == 1):
                    counter += 1
            empirical_error[i] += counter / len(train_set)

        # compute true error of every Hk(x)
        # true error  = num of misclassified points in test set / num of points

        for i in range(K):
            counter = 0
            for p2 in test_set:
                if (p2.tag == 1 and p2.final_prediction[i] == -1) or \
                        (p2.tag == -1 and p2.final_prediction[i] == 1):  # two cases of misclassification
                    counter += 1
            true_error[i] += counter / len(test_set)

    end_time = time.perf_counter()
    print(end_time - start_time)
    for i in range(K):
        print("Training Set: the empirical error of the training set for #{} rules is: {} ".format(i + 1, empirical_error[i] / iterations))
        print("Test Set: the true error of the test set for #{} rules is: {} ".format(i + 1, true_error[i] / iterations))
        print("")
        '''print("true_error avg for ", ADABOOST_ITERATIONS, " runs: ", true_error[k] / ADABOOST_ITERATIONS,
              ", empirical_error: ", empirical_error[k] / ADABOOST_ITERATIONS)'''