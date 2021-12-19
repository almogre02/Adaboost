import numpy as np
import time
import random
from adaboost import Rule, Point, adaboost, line_from_points, get_rule_prediction, update_weights

if __name__ == '__main__':
    ADABOOST_ITERATIONS = 100
    K = 8
    # read file
    f = open("../../Desktop/rectangle.txt")
    read_lines = f.read().splitlines()

    # split lines into array '1.25 2.50 1' --> [1.25, 2.50, 1]
    points_list = []
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
                rules.insert(i,
                             Rule(float(points_list[i][0]), float(points_list[i][1]),
                                  float(points_list[j][0]), float(points_list[j][1])))
    #  set for every rule the set of points tagged as +
    for r in rules:  # loop each rule
        a1, b1, c1 = line_from_points([r.x1, r.y1], [r.x2, r.y2])
        for p in points_arr:  # loop each point for each rule
            if get_rule_prediction(a1, b1, c1, p.x, p.y, np.sqrt(r.x1**2 + r.y1**2) < np.sqrt(r.x2**2 + r.y2**2)):
                r.positive_points.append(p)
            else:
                r.negative_points.append(p)
    start_time=time.perf_counter()#
    true_error, empirical_error = [0 for _ in range(K)], [0 for _ in range(K)]
    for _ in range(ADABOOST_ITERATIONS):

        # split data randomly into two halves
        random.shuffle(points_arr)
        arr_size = int(len(points_arr) / 2)
        train_data = points_arr[:arr_size]
        test_data = points_arr[arr_size:]

        # init points weights
        for ll in train_data:
            ll.weight = 1 / len(train_data)

        # execute 8 iteration of adaboost to find the 8 best rules (chosen by the biggest alpha)
        results = []
        for i in range(K):
            result = adaboost(rules, train_data)
            results.insert(i, [result, result.alpha])
            update_weights(result, train_data)

        # build Hk(x) rules
        Hk = [[results[0]],
              [results[0], results[1]],
              [results[0], results[1], results[2]],
              [results[0], results[1], results[2], results[3]],
              [results[0], results[1], results[2], results[3], results[4]],
              [results[0], results[1], results[2], results[3], results[4], results[5]],
              [results[0], results[1], results[2], results[3], results[4], results[5], results[6]],
              [results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]]

        # get Hk(x) final prediction on every point
        for p in points_arr:
            p.final_prediction = []
            for i in range(len(Hk)):
                final_prediction = 0.0
                for j in range(len(Hk[i])):
                    if p in Hk[i][j][0].positive_points:
                        prediction = 1
                    else:
                        prediction = -1
                    final_prediction += Hk[i][j][1] * prediction
                p.final_prediction.insert(i, np.sign(final_prediction))

        # compute empirical error of every Hk(x)
        # empirical error  = num of misclassified points in train set / num of points
        for i in range(K):
            counter = 0
            for p2 in train_data:
                if (p2.tag == 1 and p2.final_prediction[i] == -1) or \
                        (p2.tag == -1 and p2.final_prediction[i] == 1):
                    counter += 1
            empirical_error[i] += counter / len(train_data)

        # compute true error of every Hk(x)
        # true error  = num of misclassified points in test set / num of points

        for i in range(K):
            counter = 0
            for p2 in test_data:
                if (p2.tag == 1 and p2.final_prediction[i] == -1) or \
                        (p2.tag == -1 and p2.final_prediction[i] == 1):  # two cases of misclassification
                    counter += 1
            true_error[i] += counter / len(test_data)

    end_time=time.perf_counter()
    print(end_time-start_time)
    for k in range(K):
        print("true_error avg for ", ADABOOST_ITERATIONS, " runs: ", true_error[k] / ADABOOST_ITERATIONS,
              ", empirical_error: ", empirical_error[k] / ADABOOST_ITERATIONS)
