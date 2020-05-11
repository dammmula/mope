import math
import random
from _decimal import Decimal
from itertools import compress
from scipy.stats import f, t
import numpy
from functools import reduce
from datetime import datetime

x1min, x2min, x3min = 20, 25, 25  # задані значення Xmin
x1max, x2max, x3max = 70, 65, 35  # задані значення Xmax

x_avr_min = (x1min + x2min + x3min) / 3  # середнє значення Xmin
x_avr_max = (x1max + x2max + x3max) / 3  # середнє значення Xmax

x0_i = [(x1max + x1min) / 2, (x2max + x2min) / 2, (x3max + x3min) / 2]  # X0i
det_x_i = [(x1min - x0_i[0]), (x2min - x0_i[1]), (x3min - x0_i[2])]  #delta Xi
l = 1.73  #l=sqrt(k), де k=3

norm_plan_raw = [[-1, -1, -1],  # матриця планування при п'ятирівневому трьохфакторному експерименті з урахуванням квадратичних членів
                 [-1, +1, +1],
                 [+1, -1, +1],
                 [+1, +1, -1],
                 [-1, -1, +1],
                 [-1, +1, -1],
                 [+1, -1, -1],
                 [+1, +1, +1],
                 [-1.73, 0, 0],
                 [+1.73, 0, 0],
                 [0, -1.73, 0],
                 [0, +1.73, 0],
                 [0, 0, -1.73],
                 [0, 0, +1.73]]

natur_plan_raw = [[x1min, x2min, x3min],  # матриця планування при п'ятирівневому трьохфакторному експерименті з урахуванням квадратичних членів для
                  [x1min, x2max, x3max],  # натуральних значень факторів
                  [x1max, x2min, x3max],
                  [x1max, x2max, x3min],
                  [x1min, x2min, x3max],
                  [x1min, x2max, x3min],
                  [x1max, x2min, x3min],
                  [x1max, x2max, x3max],
                  [-l * det_x_i[0] + x0_i[0], x0_i[1], x0_i[2]],
                  [l * det_x_i[0] + x0_i[0], x0_i[1], x0_i[2]],
                  [x0_i[0], -l * det_x_i[1] + x0_i[1], x0_i[2]],
                  [x0_i[0], l * det_x_i[1] + x0_i[1], x0_i[2]],
                  [x0_i[0], x0_i[1], -l * det_x_i[2] + x0_i[2]],
                  [x0_i[0], x0_i[1], l * det_x_i[2] + x0_i[2]],
                  [x0_i[0], x0_i[1], x0_i[2]]]


def regression_equation(x1, x2, x3, coefficients, importance= [True] * 11): # рівняння регресії
    factors_array = [1, x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2]  # форма підстановки значень х
    return sum([el[0] * el[1] for el in compress(zip(coefficients, factors_array), importance)])


def func(x1, x2, x3): # задана функція
    coefficients = [0.6, 4, 2.8, 4.7, 3.1, 0.4, 5.4, 5.7, 0.1, 8.8, 0.1]
    return regression_equation(x1, x2, x3, coefficients) # формування рівняння регресії


def generate_factors_table(raw_array): # створення таблиці факторів
    raw_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]] + list(
        map(lambda x: x ** 2, row)) for row in raw_array]
    return list(map(lambda row: list(map(lambda el: round(el, 3), row)), raw_list))


def generate_y(m, factors_table): # створення у
    return [[round(func(row[0], row[1], row[2]) + random.randint(-5, 5), 3) for _ in range(m)]
            for row in factors_table]


def print_matrix(m, N, factors, y_vals, additional_text=":"): # виведення матриці
    labels_table = list(map(lambda x: x.ljust(10),
                            ["x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"] + [
                                "y{}".format(i + 1) for i in range(m)]))
    rows_table = [list(factors[i]) + list(y_vals[i]) for i in range(N)]
    print("\nМатриця планування" + additional_text)
    print(" ".join(labels_table))
    print("\n".join([" ".join(map(lambda j: "{:<+10}".format(j), rows_table[i])) for i in range(len(rows_table))]))
    print("\t")


def print_equation(coefficients, importance=[True] * 11): # виведення рівняння
    x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    coefficients_to_print = list(compress(coefficients, importance))
    equation = " ".join(
        ["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), coefficients_to_print)), x_i_names)])
    print("Рівняння регресії: y = " + equation)


def set_factors_table(factors_table): # таблиця факторів
    def x_i(i):
        with_null_factor = list(map(lambda x: [1] + x, generate_factors_table(factors_table)))
        res = [row[i] for row in with_null_factor]
        return numpy.array(res)
    return x_i


def m_ij(*arrays): # Mij
    return numpy.average(reduce(lambda accum, el: accum * el, list(map(lambda el: numpy.array(el), arrays))))


def find_coefficients(factors, y_vals): # знаходження коефіцієнтів
    x_i = set_factors_table(factors) # Xi
    coefficients = [[m_ij(x_i(column), x_i(row)) for column in range(11)] for row in range(11)] # коефіцієнти рівняння
    y_numpy = list(map(lambda row: numpy.average(row), y_vals))
    free_values = [m_ij(y_numpy, x_i(i)) for i in range(11)] # вільні члени
    beta_coefficients = numpy.linalg.solve(coefficients, free_values) # коефіцієнти бета
    return list(beta_coefficients)


def cochran_criteria(m, N, y_table): # критерій Кохрена

    def get_cochran_value(f1, f2, q):

        partResult1 = q / f2
        params = [partResult1, f1, (f2 - 1) * f1]
        fisher = f.isf(*params)
        result = fisher / (fisher + (f2 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()


    print("Перевірка рівномірності дисперсій за критерієм Кохрена: m = {}, N = {}".format(m, N))
    start_time1 = datetime.now()
    y_variations = [numpy.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation / sum(y_variations)
    f1 = m - 1 # визначення степенів свободи
    f2 = N
    p = 0.95  # ймовірність
    q = 1 - p  # рівень значимості
    gt = get_cochran_value(f1, f2, q) # табличне значення коефіцієнта Кохрена
    print("Gp = {}; Gt = {}; f1 = {}; f2 = {}; q = {:.2f}".format(gp, gt, f1, f2, q))

    if gp < gt:
        print("Gp < Gt => дисперсії рівномірні - все правильно")
        print("Час виконання перевірки: %s секунд" % (datetime.now() - start_time1))

        return True
    else:
        print("Gp > Gt => дисперсії нерівномірні - треба ще експериментів")
        return False


def student_criteria(m, N, y_table, beta_coefficients): # критерій Ст'юдента
    def get_student_value(f3, q):
        return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()

    print("\nПеревірка значимості коефіцієнтів регресії за критерієм Стьюдента: m = {}, N = {} ".format(m, N))
    start_time2 = datetime.now()
    average_variation = numpy.average(list(map(numpy.var, y_table)))
    x_i = set_factors_table(natural_plan)
    variation_beta_s = average_variation / N / m
    standard_deviation_beta_s = math.sqrt(variation_beta_s)
    t_i = numpy.array([abs(beta_coefficients[i]) / standard_deviation_beta_s for i in range(len(beta_coefficients))])
    f3 = (m - 1) * N # степерні свободи
    q = 0.05 # рівень значимості (1-р)
    t_our = get_student_value(f3, q) # табличне значення критерія Ст'юдента
    importance = [True if el > t_our else False for el in list(t_i)]

    print("Оцінки коефіцієнтів βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
    print("Коефіцієнти ts:         " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    print("f3 = {}; q = {}; tтабл = {}".format(f3, q, t_our))
    beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
    importance_to_print = ["важливий" if i else "неважливий" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
    print(*to_print, sep="; ")
    print("Час виконання перевірки: %s секунд" % (datetime.now() - start_time2))
    print_equation(beta_coefficients, importance)
    return importance


def fisher_criteria(m, N, d, x_table, y_table, b_coefficients, importance): # критерій Фішера
    def get_fisher_value(f3, f4, q):
        return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()

    start_time = datetime.now()
    f3 = (m - 1) * N # число степенів свободи
    f4 = N - d
    q = 0.05 # рівень значимості
    theoretical_y = numpy.array([regression_equation(row[0], row[1], row[2], b_coefficients) for row in x_table]) # теоретичне у
    average_y = numpy.array(list(map(lambda el: numpy.average(el), y_table))) # середнє у
    s_ad = m / (N - d) * sum((theoretical_y - average_y) ** 2)
    y_variations = numpy.array(list(map(numpy.var, y_table)))
    s_v = numpy.average(y_variations)
    f_p = float(s_ad / s_v) # експерементальне значення
    f_t = get_fisher_value(f3, f4, q)  # табличне значення коефіцієнта Фішера
    theoretical_values_to_print = list(zip(map(lambda x: "x1 = {0[1]:<10} x2 = {0[2]:<10} x3 = {0[3]:<10}"
                                               .format(x), x_table), theoretical_y))
    print("\nПеревірка адекватності моделі за критерієм Фішера: m = {}, N = {} для таблиці y_table".format(m, N))
    print("Теоретичні значення y для різних комбінацій факторів:")
    print("\n".join(["{arr[0]}: y = {arr[1]}".format(arr=el) for el in theoretical_values_to_print]))
    print("Fp = {}, Ft = {}".format(f_p, f_t))
    print("Fp < Ft => модель адекватна" if f_p < f_t else "Fp > Ft => модель неадекватна")

    print("Час виконання перевірки: %s секунд" % (datetime.now() - start_time))
    return True if f_p < f_t else False


m = 3 # кількість повторів кожної комбінації
N = 15 # кількість рівнів
natural_plan = generate_factors_table(natur_plan_raw)
y_arr = generate_y(m, natur_plan_raw)
while not cochran_criteria(m, N, y_arr): # збільшення m
    m += 1
    y_arr = generate_y(m, natural_plan)
print_matrix(m, N, natural_plan, y_arr, " для натуралізованих факторів:")
coefficients = find_coefficients(natural_plan, y_arr)
print_equation(coefficients)
importance = student_criteria(m, N, y_arr, coefficients)
d = len(list(filter(None, importance)))
fisher_criteria(m, N, d, natural_plan, y_arr, coefficients, importance)